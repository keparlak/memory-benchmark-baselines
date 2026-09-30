"""Deney 5 — Mem0 kolu.

Mem0 kendi LLM'ini cagirir (fact extraction + update kararı). Onu da
NIM'e yonlendiriyoruz ki tum kollar ayni modeli kullansin. Embedder
lokal tutulur: 40 rpm'lik butce yalnizca LLM cagrilarina gitsin.

ONEMLI: Mem0'in stored_v2'si, urunun IC DURUMUNU sorgular (get_all).
Bu write path olcumudur. read() ise search() ciktisidir -- read path.
Ayrim bu ikisinde; Asama 1'deki ayni tanim.
"""
import os
import random
import shutil
import tempfile
import time
import uuid
import warnings

warnings.filterwarnings("ignore")

# Mem0 kendi telemetri/migration deposunu da qdrant ile aciyor ve o depo
# ~/.mem0 altinda PAYLASILIYOR -- her kosuda yeni Memory nesnesi acmak
# "Storage folder ... already accessed" hatasi veriyor. Telemetriyi kapat.
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import deney5_llm as L
from deney5_kollar import Memory, SIG

# torch kurulumu bu makinede bozuk (shm.dll yuklenemiyor), o yuzden
# sentence-transformers yerine fastembed: ONNX tabanli, torch gerektirmez.
# Embedder lokal kalir -> 40 rpm butcesi yalnizca LLM'e gider.
RETRY_STATS = {"mem0_llm_retries": 0, "mem0_llm_failed": 0}
SEARCH_TOP_K = 20   # Deney 5'te fiilen calisan deger (limit=12 yok sayilmisti)
VERBOSE = True   # hatalari sessizce yutma: sifir sonucu kurulum hatasi olabilir
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # 384 boyut, fastembed varsayilani


def _config(collection, path):
    return {
        "llm": {"provider": "openai", "config": {
            "model": L.MODEL,
            "openai_base_url": L.BASE,
            "api_key": L.KEY,
            "temperature": 0.0,
            "max_tokens": 1200,
        }},
        "embedder": {"provider": "fastembed", "config": {
            "model": EMBED_MODEL,
        }},
        # her kol kendi izole deposunu alir. on_disk=False tek basina
        # yetmiyor: qdrant lokal modda paylasilan bir klasoru kilitliyor
        # ("Storage folder /tmp/qdrant is already accessed by another
        # instance"). Benzersiz dizin + close() ile cozulur.
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": collection,
            "path": path,
            "on_disk": False,
            "embedding_model_dims": 384,
        }},
    }


def patch_thinking_off():
    """Mem0'in kendi LLM cagrilarinda da reasoning'i kapat.

    Mem0'in config'i bu parametreyi almiyor, ama OpenAI SDK extra_body
    destekliyor. Yama olmadan Mem0'in fact-extraction adimi BOS content
    aliyor ve "Expecting value: line 1 column 1" ile hicbir sey saklamiyordu
    -- urun hatasi gibi gorunen, aslinda kurulum hatasi olan durum.

    Yama idempotent: birden fazla Mem0Arm olusturulsa da bir kez uygulanir.
    """
    from openai.resources.chat import completions as _c
    if getattr(_c.Completions.create, "_d5_patched", False):
        return
    orig = _c.Completions.create

    def create(self, *a, **kw):
        eb = dict(kw.get("extra_body") or {})
        eb.setdefault("chat_template_kwargs", {"thinking": False})
        kw["extra_body"] = eb
        kw.setdefault("max_tokens", 1500)
        # Servis dalgalaniyor (503 "temporarily overloaded"). Retry olmadan
        # bu kayiplar Mem0'in skoruna yaziliyordu -- urunun degil altyapinin
        # hatasi. Ilk tam kosuda 8 yazma bu yuzden kaybolmustu.
        # 8 deneme + uzun backoff: 20 rpm'e dusurmek 429'lari bitirmedi,
        # demek ki saglayicinin siniri sadece rpm degil (burst/token de
        # olabilir). 429 gecici oldugu icin cozum beklemek: yazma kaybi
        # olcumu bozuyordu, gecikme bozmuyor.
        last = None
        for i in range(8):
            # ONEMLI: Mem0'in cagrilari da ayni hiz siniridan gecmeli.
            # Gecmedigi surumde ajan cagrilari (34 rpm) + Mem0 cagrilari
            # birlikte hesabin 40 rpm sinirini asiyordu; 429'lar retry
            # uretiyor, retry'lar yeni 429 uretiyor ve kosu ilerlemiyordu.
            L._limiter.wait()
            try:
                return orig(self, *a, **kw)
            except Exception as e:
                m = str(e)
                if any(c in m for c in ("503", "429", "500", "502", "504",
                                        "overloaded", "timed out", "timeout")):
                    last = e
                    RETRY_STATS["mem0_llm_retries"] += 1
                    time.sleep(min(2 ** i + random.random(), 45))
                    continue
                raise
        RETRY_STATS["mem0_llm_failed"] += 1
        raise last

    create._d5_patched = True
    _c.Completions.create = create


def _event_text(ev):
    """Olayi Mem0'a verilecek duz metne cevir. render() ile ayni bicim."""
    if ev["type"] == "tool_update":
        return ev["text"]
    c = ev["call"]
    args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
    return f"called {c['tool']}({args})"


class Mem0Arm(Memory):
    """Gercek urun. Olaylar partiler halinde yazilir.

    batch=1 en sadik olurdu (her olay ayri add), ama 40 rpm limitinde
    40 kosu x 60 olay = 2400 add cagrisi saatler surer. batch dengeyi
    kurar; kacinilmaz bir odun ve rapora yazilir.
    """
    name = "B mem0"

    _shared = None          # tum kollar tek Mem0 ornegi paylasir
    _shared_dir = None

    @classmethod
    def _get_shared(cls):
        """Tek Memory ornegi; izolasyon user_id ile saglanir.

        Her kosuda yeni Memory acmak hem pahali hem kilit catismasi
        uretiyordu. Mem0 zaten user_id ile filtreliyor, o yuzden kosu
        basina ayri user_id yeterli ve daha sadik: gercekte de tek bir
        hafiza servisi coklu oturuma hizmet eder.
        """
        if cls._shared is None:
            patch_thinking_off()
            from mem0 import Memory as Mem0Memory
            cls._shared_dir = tempfile.mkdtemp(prefix="d5_qdrant_")
            cls._shared = Mem0Memory.from_config(
                _config("d5_shared", cls._shared_dir))
        return cls._shared

    @classmethod
    def shutdown(cls):
        if cls._shared is not None:
            try:
                cls._shared.vector_store.client.close()
            except Exception:
                pass
            shutil.rmtree(cls._shared_dir, ignore_errors=True)
            cls._shared = None

    def __init__(self, batch=10, user_id=None):
        self.m = self._get_shared()
        self.coll = "d5_shared"
        self.user_id = user_id or f"ep_{uuid.uuid4().hex[:10]}"
        self.batch = batch
        self._buf = []
        self._steps = []
        self.add_calls = 0
        self.errors = 0
        self.last_error = None
        self.slot_log = []

    # --- yazma ---
    def _flush(self):
        if not self._buf:
            return
        msgs = [{"role": "user", "content": t} for t in self._buf]
        try:
            res = self.m.add(msgs, user_id=self.user_id)
            self.add_calls += 1
            # Kartik'in sorusu: "hangi hafiza slotuna dokundu?"
            # Mem0 her yazma icin ADD/UPDATE/DELETE/NONE donduruyor.
            items = res.get("results", []) if isinstance(res, dict) else (res or [])
            for it in items:
                self.slot_log.append({
                    "event": it.get("event"),
                    "id": it.get("id"),
                    "memory": (it.get("memory") or "")[:120],
                })
        except Exception as e:
            self.errors += 1
            self.last_error = f"add: {type(e).__name__}: {e}"
            if VERBOSE:
                print("  ADD HATASI:", self.last_error[:300])
        self._buf = []

    def write(self, ev):
        self._steps.append(ev)
        self._buf.append(_event_text(ev))
        if len(self._buf) >= self.batch:
            self._flush()

    # --- okuma ---
    def _as_events(self, texts):
        """Mem0 duz metin dondurur; olcum icin olay bicimine geri cevir.

        Bir metin guncel imzayi tasiyorsa tool_update olayi gibi davranir.
        Tasimiyorsa cagri kaydi olarak gecer. Bu cevrim olcumu gevsetmez:
        imza ancak TAM eslesirse v2 sayilir.
        """
        out = []
        for i, t in enumerate(texts):
            m = SIG.search(t or "")
            if m:
                tool = m.group(1)
                fields = {}
                for part in m.group(2).split(","):
                    if ":" in part:
                        n, ty = part.split(":", 1)
                        fields[n.strip()] = ty.strip()
                out.append({"step": i, "type": "tool_update", "tool": tool,
                            "schema": {tool: fields}, "text": t})
            else:
                out.append({"step": i, "type": "note", "text": t})
        return out

    def read(self, tool):
        self._flush()
        try:
            # mem0 2.1.0: user_id artik filters icinde geciyor.
            # DIKKAT: parametrenin adi top_k. Eskiden limit=12 veriliyordu ve
            # **kwargs'a dusup SESSIZCE yok sayiliyordu -> varsayilan top_k=20
            # calisti. Deney 5 bu yuzden ajana 12 degil 20 kayit verdi.
            # Karsilastirilabilirlik icin 20'yi acikca sabitliyoruz.
            r = self.m.search(query=f"{tool} current schema arguments",
                              filters={"user_id": self.user_id}, top_k=SEARCH_TOP_K)
        except Exception as e:
            self.errors += 1
            self.last_error = f"search: {type(e).__name__}: {e}"
            if VERBOSE:
                print("  SEARCH HATASI:", self.last_error[:300])
            return []
        items = r.get("results", r) if isinstance(r, dict) else r
        return self._as_events([i.get("memory", "") for i in (items or [])])

    def stored_events(self):
        self._flush()
        try:
            # DIKKAT: parametrenin adi top_k (varsayilan 20). Eskiden
            # limit=10000 veriliyordu ve sessizce yok sayiliyordu: get_all
            # ~50 kaydin yalnizca 20'sini dondurdu. Deney 5'in stored
            # sutunlari bu yuzden deponun ~%40'lik kesitinden olculdu ve
            # "stored < retrieved" celiskisi buradan geldi.
            r = self.m.get_all(filters={"user_id": self.user_id}, top_k=100000)
        except Exception as e:
            self.errors += 1
            self.last_error = f"get_all: {type(e).__name__}: {e}"
            if VERBOSE:
                print("  GET_ALL HATASI:", self.last_error[:300])
            return []
        items = r.get("results", r) if isinstance(r, dict) else r
        return self._as_events([i.get("memory", "") for i in (items or [])])


    def search_texts(self, query, top_k):
        """Ham arama: siralanmis kayit metinleri (Deney 6 sira analizi icin)."""
        self._flush()
        r = self.m.search(query=query, filters={"user_id": self.user_id}, top_k=top_k)
        items = r.get("results", r) if isinstance(r, dict) else r
        return [i.get("memory", "") for i in (items or [])]

    def all_texts(self):
        """Deponun TAMAMI (top_k buyuk). Deney 6'da depo buyuklugu icin."""
        self._flush()
        r = self.m.get_all(filters={"user_id": self.user_id}, top_k=100000)
        items = r.get("results", r) if isinstance(r, dict) else r
        return [i.get("memory", "") for i in (items or [])]

    def close(self):
        """Bu kolun kayitlarini sil. Paylasilan ornek ayakta kalir."""
        try:
            self.m.delete_all(filters={"user_id": self.user_id})
        except Exception:
            pass


if __name__ == "__main__":
    # tek kosuluk duman testi: config ayakta mi, kac cagri yakiyor
    import time
    import random
    from deney5_korpus import make_episode

    rng = random.Random(20260922)
    ep = make_episode(0, rng, n_steps=20)
    tool = ep["mutation"]["tool"]
    print(f"kosu: {len(ep['events'])} olay, mutasyon={ep['mutation']}")

    t = time.time()
    arm = Mem0Arm(batch=10)
    for ev in ep["events"]:
        arm.write(ev)
    stored = arm.stored_v2(tool, set(ep["v2"][tool]))
    ctx = arm.read(tool)
    retrieved = any(e["type"] == "tool_update" and e["tool"] == tool
                    and set(e["schema"][tool]) == set(ep["v2"][tool]) for e in ctx)

    print(f"sure={time.time()-t:.0f}s  add_calls={arm.add_calls}  errors={arm.errors}")
    print(f"stored_v2={stored}  retrieved_v2={retrieved}  ctx={len(ctx)} kayit")
    print(f"llm stats={L.STATS}")
    for e in ctx[:6]:
        print("   -", (e.get("text") or "")[:90])
