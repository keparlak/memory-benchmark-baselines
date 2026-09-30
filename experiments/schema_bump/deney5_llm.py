"""Deney 5 — NVIDIA NIM istemcisi.

Hesap limiti 40 rpm (profil menusunde yaziyor), o yuzden istekler kendi
icinde hizlandirilir. 503/429 gecici olabiliyor; ustel geri cekilme ile
tekrar denenir. Basarisiz cagri sessizce atlanmaz, sayilir -- deneyde
"cevap alamadik" ile "yanlis cevap" ayri seylerdir.
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

MODEL = "nvidia/nemotron-3-super-120b-a12b"
RPM = 20          # 40'in cok altinda.
#
# 34 ile kosuldugunda Mem0'in yazmalari 429 aliyordu: kosu basina ~52 cagri
# gidiyor ve RETRY'lar da hesabin 40 rpm kotasindan yeniyor, toplam siniri
# asiyordu. Bir kosuda Mem0 61 olaydan sadece 1 kayit tutabildi -- olculen
# sey urunun davranisi degil, saglayicinin hiz siniri oldu.
# 20 rpm kosuyu ~2x uzatiyor ama olcumu temiz birakiyor.


# .env'den okunur ama ORTAMA KONMAZ. Sebep: mem0'in OpenAILLM sinifi
#
#     if os.environ.get("OPENROUTER_API_KEY"):   # Use OpenRouter
#         self.client = OpenAI(api_key=..., base_url=...openrouter...)
#     else:
#         api_key = self.config.api_key or ...
#
# yani ortamda bu degisken varsa ACIKCA verilen config'i yok sayip
# OpenRouter'a gidiyor. Bir kosuyu boyle kaybettik: Mem0'a NVIDIA
# verdigimiz halde tum yazmalar OpenRouter'a gitti ve 402 aldi.
SECRETS = {}


def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            SECRETS[k] = v
            if k.startswith("OPENROUTER"):
                continue            # ortama koyma - mem0'i kacirir
            os.environ.setdefault(k, v)


_load_env()
KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


class _Limiter:
    """Basit token-bucket: dakikada en fazla RPM istek."""

    def __init__(self, rpm):
        self.min_gap = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.monotonic()
            self.next_at = now + self.min_gap


_limiter = _Limiter(RPM)

STATS = {"calls": 0, "retries": 0, "failed": 0, "tokens": 0}
_stats_lock = threading.Lock()


class LLMError(RuntimeError):
    pass


def chat(messages, model=MODEL, max_tokens=800, temperature=0.0, attempts=5,
         thinking=False):
    """Tek bir sohbet cagrisi.

    thinking=False varsayilan: bu model reasoning yapiyor ve uzun promptlarda
    dusunme butcesini tuketip content'i BOS birakabiliyor. Olcum (8 kosu):
        thinking=True,  max_tokens=1500 -> 8/8 dogru, 60s
        thinking=False, max_tokens=800  -> 8/8 dogru, 23s
    Dogruluk kaybi yok, 2.6x hiz. Bos content ayrica Mem0'in fact-extraction
    adimini da kiriyordu.
    """
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if not thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    payload = json.dumps(body).encode()
    last = None
    for i in range(attempts):
        _limiter.wait()
        req = urllib.request.Request(
            BASE + "/chat/completions", data=payload, method="POST",
            headers={"Authorization": f"Bearer {KEY}",
                     "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            msg = r["choices"][0]["message"]
            with _stats_lock:
                STATS["calls"] += 1
                STATS["tokens"] += r.get("usage", {}).get("total_tokens") or 0
            return msg.get("content") or ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                with _stats_lock:
                    STATS["retries"] += 1
                time.sleep(min(2 ** i + random.random(), 30))
                continue
            raise LLMError(f"{last}: {e.read()[:200].decode(errors='replace')}")
        except Exception as e:                      # timeout vb.
            last = type(e).__name__
            with _stats_lock:
                STATS["retries"] += 1
            time.sleep(min(2 ** i + random.random(), 30))
    with _stats_lock:
        STATS["failed"] += 1
    raise LLMError(f"{attempts} denemede basarisiz: {last}")


def parse_call(text):
    """Modelin ciktisindan tek bir {"tool":..,"args":{..}} nesnesi cikar.

    Model reasoning yapiyor ve bazen JSON'u ``` icine aliyor; ikisini de
    tolere ederiz. Cikarilamazsa None -- bu "yanlis cagri" degil,
    "cagri uretilemedi" olarak sayilir.
    """
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        for part in s.split("```"):
            if "{" in part and "tool" in part:
                s = part
                break
    s = s.replace("json\n", "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:
        return None
    if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict):
        return obj
    return None


if __name__ == "__main__":
    SYS = ("You are an agent that calls tools. Use ONLY the CURRENT tool schema. "
           "Output exactly one JSON object: {\"tool\":..., \"args\":{...}} "
           "and nothing else.")
    FULL = """Log:
  create_ticket(title="login-401", body="cannot sign in", priority="high")
  create_ticket(title="export-timeout", body="csv export hangs", priority="low")
[tool_update] create_ticket: 'title' renamed to 'subject'; 'priority' removed,
replaced by 'severity' (int).
Current signature -> create_ticket(subject: str, body: str, severity: int)
  assign_owner(ticket_id="T-12", owner="ayse")

Task: open a ticket for "webhook retries exhausted"."""
    LOSSY = """Log:
  create_ticket(title="login-401", body="cannot sign in", priority="high")
  create_ticket(title="export-timeout", body="csv export hangs", priority="low")
  assign_owner(ticket_id="T-12", owner="ayse")

Task: open a ticket for "webhook retries exhausted"."""

    for label, ctx in (("v2 BAGLAMDA VAR", FULL), ("v2 KAYIP", LOSSY)):
        t = time.time()
        call = parse_call(chat([{"role": "system", "content": SYS},
                                {"role": "user", "content": ctx}]))
        args = sorted(call["args"]) if call else None
        ok = args == ["body", "severity", "subject"]
        print(f"{label:18} {time.time()-t:5.1f}s  "
              f"{'v2-UYUMLU' if ok else 'v2-DEGIL '}  args={args}")
    print("stats:", STATS)
