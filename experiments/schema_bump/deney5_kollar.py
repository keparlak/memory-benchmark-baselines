"""Deney 5 — hafiza kollari ve attribution olcumu. LLM yok.

Her kol bir hafiza arayuzu sunar:
    write(events)  -> kosu boyunca olaylari yaz
    read(tool)     -> o tool icin baglam dondur

Ajan Asama 1'de deterministiktir: baglamda buldugu EN SON sema tanimina gore
cagri uretir. Gercek bir LLM'in yapacagi sey de budur -- baglamda ne varsa ona
gore cagirir. Boylece olculen sey ajanin zekasi degil, hafizanin ona ne
verdigidir. Tam olarak istedigimiz ayrim.

Tasarim: DENEY5-tasarim-schema-bump.md
"""
import json
import math
import re
from collections import Counter

from deney5_korpus import validate_call, build_corpus

# Imza yakalayici. ONEK ARAMAZ -- bilerek.
#
# Ilk surum r"Guncel imza -> (\w+)\(([^)]*)\)" idi ve Mem0'i haksiz yere
# sifirliyordu: Mem0 fact-extraction yaparken metni kendi cumlesiyle yeniden
# yaziyor, benim onekim kayboluyor, ama IMZA tam olarak duruyor:
#
#   "...replaced with 'severity' (int), updating the function signature to
#    create_ticket(title: str, body: str, severity: int)"
#
# Onek arayan olcut, bilgiyi saklayan bir urunu "hic saklamamis" gosterir.
# Bu yuzden olcut icerige bakar: tool adi + icinde en az bir "ad: tip" cifti
# olan parantez. Gevsetme yanlis pozitif uretmez, cunku stored_v2 alan
# kumesinin v2 ile TAM eslesmesini arar; v1 imzasi v2 sayilmaz.
SIG = re.compile(r"\b(\w+)\(\s*([^)]*?:[^)]*?)\)")


# --- hafiza kollari -------------------------------------------------------

class Memory:
    """Ortak arayuz.

    stored_v2 write path'i, read() read path'i olcer. Ayrim bu ikisinde:
    bilgi hic saklanmadiysa yazma kaybetmistir; saklandi ama gelmiyorsa
    okuma kaybetmistir.
    """

    def stored_events(self):
        raise NotImplementedError

    def stored_v2(self, tool, v2_fields):
        """KATI olcut: hafizada v2 imzasi TAM olarak duruyor mu?

        Imza biciminde saklamayan ama bilgiyi baska sekilde tutan bir urunu
        bu olcut sifirlar. Mem0'da tam bu oldu: validity %47.5 iken bu olcut
        %7.5 gosterdi -- ajan dogru cagriyi uretebiliyorsa bilgi bir yerde
        duruyor demektir. Bu yuzden asagidaki gevsek olcut de raporlanir.
        """
        return any(e.get("type") == "tool_update" and e.get("tool") == tool
                   and set(e["schema"][tool]) == v2_fields
                   for e in self.stored_events())

    def stored_v2_loose(self, tool, v2_only, v1_only):
        """GEVSEK olcut: v2'ye ozgu alan adlari hafizada geciyor mu?

        v2_only = v2'de olup v1'de olmayan alanlar (ornegin 'severity').
        Bir kayitta gecmesi yeterli -- cagri kaydi da olabilir, cumle de.
        Ajanin ogrenebilecegi her iz sayilir.

        Katı olcut "urun bilgiyi BENIM bicimimde sakladi mi" diye sorar;
        gevsek olcut "bilgi herhangi bir bicimde orada mi" diye. Ikisinin
        arasindaki fark, urunun bicimi ile olcutun biciminin uyusmadigi
        yerdir ve rapor edilmeye deger.
        """
        if not v2_only:
            return None                     # bu mutasyonda yeni alan yok
        blob = " ".join((e.get("text") or "") for e in self.stored_events())
        return all(f in blob for f in v2_only)


class NaiveAppend(Memory):
    """Konsolidasyon yok, her sey saklanir, hepsi geri verilir.

    Tavan kolu: hicbir sey kaybetmez. Bir hafiza urunu bunun altinda
    kaliyorsa, kaybi ekledigi katman yaratmistir.
    """
    name = "naive-append (hepsini sakla)"

    def __init__(self):
        self.store = []

    def write(self, ev):
        self.store.append(ev)

    def read(self, tool):
        return list(self.store)

    def stored_events(self):
        return self.store


class TfidfRag(Memory):
    """Benim MCC tabanimin bu teste tasinmis hali: top-k benzerlik."""
    name = "TF-IDF RAG (top-k)"

    def __init__(self, k=6):
        self.k = k
        self.docs = []
        self.df = Counter()

    @staticmethod
    def _tok(ev):
        if ev["type"] == "tool_update":
            return ev["text"].lower().replace("(", " ").replace(")", " ").split()
        c = ev["call"]
        return (c["tool"] + " " + " ".join(c["args"])).lower().split()

    def write(self, ev):
        toks = self._tok(ev)
        self.docs.append((ev, Counter(toks)))
        for t in set(toks):
            self.df[t] += 1

    def read(self, tool):
        if not self.docs:
            return []
        n = len(self.docs)
        q = Counter(tool.lower().split("_"))
        scored = []
        for ev, tf in self.docs:
            s = 0.0
            for t, c in q.items():
                if t in tf:
                    s += c * tf[t] * math.log(1 + n / (1 + self.df[t]))
            scored.append((s, ev))
        scored.sort(key=lambda x: (-x[0], x[1]["step"]))
        return [ev for _, ev in scored[: self.k]]

    def stored_events(self):
        return [ev for ev, _ in self.docs]


class SupersessionPolicy(Memory):
    """Deterministik politika: en son sema tanimi kazanir.

    FactConsolidation'da ayni politika kapsam ici %96.6-100 vermisti.
    Burada da tavan vermeli. Vermezse politika degil kurulum hatalidir.
    """
    name = "deterministic supersession"

    def __init__(self):
        self.latest = {}   # tool -> son tool_update olayi
        self.calls = []

    def write(self, ev):
        if ev["type"] == "tool_update":
            self.latest[ev["tool"]] = ev
        else:
            self.calls.append(ev)

    def read(self, tool):
        out = []
        if tool in self.latest:
            out.append(self.latest[tool])
        out += [c for c in self.calls if c["call"]["tool"] == tool][-3:]
        return out

    def stored_events(self):
        return list(self.latest.values()) + self.calls


class LossyConsolidate(Memory):
    """Kontrol kolu: LLM tabanli hafiza urunlerinin yaptigi seyin karikaturu.

    Ayni tool icin gelen kayitlari tek bir "ozet"e indirger ve ozeti
    ILK gordugu sekle gore sabitler -- yeni gelen bilgiyi eski kalibin
    icine sikistirir. Mem0/Cognee'nin rewrite adiminin en kotu hali.

    Bu bir iddia degil, bir referans egrisi: write path bozuldugunda
    olcumlerin nasil gorundugunu gosterir.
    """
    name = "lossy-consolidate (write path bozuk)"

    def __init__(self):
        self.summary = {}

    def write(self, ev):
        if ev["type"] == "tool_update":
            t = ev["tool"]
            if t not in self.summary:
                self.summary[t] = ev
            # zaten bir ozet varsa yeni sema atilir -- kirli write budur
        else:
            t = ev["call"]["tool"]
            self.summary.setdefault(t, ev)

    def read(self, tool):
        return [self.summary[tool]] if tool in self.summary else []

    def stored_events(self):
        return list(self.summary.values())


# --- deterministik ajan ---------------------------------------------------

def agent_call(tool, context, rng_seed=0):
    """Baglamdaki en son sema tanimina gore cagri uret.

    Oncelik: (1) tool_update imzasi, (2) en son gorulen cagrinin arg sekli.
    Hicbiri yoksa cagri uretilemez.
    """
    shape = None
    for ev in context:                      # tool_update varsa onu al
        if ev["type"] == "tool_update" and ev["tool"] == tool:
            m = SIG.search(ev["text"])
            if m and m.group(1) == tool:
                shape = {}
                for part in m.group(2).split(","):
                    if ":" in part:
                        n, t = part.split(":", 1)
                        shape[n.strip()] = t.strip()
    if shape is None:                       # yoksa son cagriyi taklit et
        for ev in context:
            if ev["type"] == "call" and ev["call"]["tool"] == tool:
                shape = {n: ("int" if isinstance(v, int) else "str")
                         for n, v in ev["call"]["args"].items()}
    if shape is None:
        return None
    args = {}
    for i, (n, t) in enumerate(shape.items()):
        args[n] = (i + 1) if t == "int" else f"val-{i}"
    return {"tool": tool, "args": args}


# --- olcum ----------------------------------------------------------------

def v1_only_fields(ep):
    """v1'de olup v2'de olmayan alan adlari -- bayat yazinin parmak izi."""
    t = ep["mutation"]["tool"]
    return set(ep["v1"][t]) - set(ep["v2"][t])


def run_arm(corpus, make_mem):
    """Tek bir kolu tum korpusta kosur.

    Iki bagimsiz olcum:
      stored_v2    -> v2 tanimi hafizanin ic durumunda duruyor mu   (WRITE)
      retrieved_v2 -> read() ile donen baglamda v2 tanimi var mi    (READ)

    Ilk surumde "stale_write"i "baglamda v1 alanlari geciyor mu" diye
    olcmustum; yanlisti. Gecmis cagrilarin v1 olmasi kirlilik degil tarihtir.
    naive-append %100 dogru cagri uretirken %78 kirli gorunuyordu -- metrigin
    bozuk oldugunun isareti buydu.
    """
    stats = {"valid": 0, "total": 0, "stored_v2": 0, "retrieved_v2": 0,
             "no_call": 0, "by_mutation": {}, "attribution": Counter()}

    for ep in corpus:
        mem = make_mem()
        for ev in ep["events"]:
            mem.write(ev)

        tool = ep["mutation"]["tool"]
        kind = ep["mutation"]["kind"]
        v2_fields = set(ep["v2"][tool])

        stored = mem.stored_v2(tool, v2_fields)          # WRITE path
        ctx = mem.read(tool)
        retrieved = any(e["type"] == "tool_update" and e["tool"] == tool
                        and set(e["schema"][tool]) == v2_fields
                        for e in ctx)                     # READ path

        call = agent_call(tool, ctx)
        ok = False
        if call is None:
            stats["no_call"] += 1
        else:
            ok, _ = validate_call(call, ep["v2"])

        stats["total"] += 1
        stats["valid"] += int(ok)
        stats["stored_v2"] += int(stored)
        stats["retrieved_v2"] += int(retrieved)
        b = stats["by_mutation"].setdefault(kind, [0, 0])
        b[0] += int(ok)
        b[1] += 1
        stats["attribution"][
            ("v2 saklandi" if stored else "v2 SAKLANMADI",
             "v2 getirildi" if retrieved else "v2 getirilmedi")] += 1

    return stats


def pct(a, b):
    return 100.0 * a / b if b else 0.0


if __name__ == "__main__":
    corpus = build_corpus()
    arms = [NaiveAppend, TfidfRag, SupersessionPolicy, LossyConsolidate]

    rows = []
    details = {}
    for cls in arms:
        st = run_arm(corpus, cls)
        rows.append((cls.name,
                     pct(st["valid"], st["total"]),
                     pct(st["stored_v2"], st["total"]),
                     pct(st["retrieved_v2"], st["total"])))
        details[cls.name] = st

    w = max(len(r[0]) for r in rows)
    print(f"{'kol'.ljust(w)}  call_validity@v2   stored_v2   retrieved_v2")
    print(f"{''.ljust(w)}  {'(ana skor)':>16}   {'(WRITE)':>9}   {'(READ)':>12}")
    print("-" * (w + 46))
    for name, v, st_, rt in rows:
        print(f"{name.ljust(w)}  {v:15.1f}%  {st_:9.1f}%  {rt:12.1f}%")

    print("\n--- mutasyon tipine gore call_validity@v2 ---")
    kinds = sorted(details[arms[0].name]["by_mutation"])
    print("kol".ljust(w) + "  " + "  ".join(k.rjust(12) for k in kinds))
    for cls in arms:
        st = details[cls.name]
        cells = []
        for k in kinds:
            ok, n = st["by_mutation"][k]
            cells.append(f"{pct(ok, n):11.1f}%")
        print(cls.name.ljust(w) + "  " + "  ".join(cells))

    print("\n--- attribution tablosu ---")
    for cls in arms:
        st = details[cls.name]
        print(f"\n{cls.name}:")
        for (wr, rd), c in sorted(st["attribution"].items()):
            print(f"    {wr:12} | {rd:15} : {c:4}  ({pct(c, st['total']):.1f}%)")

    with open("deney5_sonuc_asama1.json", "w", encoding="utf-8") as f:
        json.dump({n: {k: (dict((f"{a}|{b}", c) for (a, b), c in v.items())
                            if k == "attribution" else v)
                       for k, v in s.items()}
                   for n, s in details.items()}, f, ensure_ascii=False, indent=2)


class TfidfRecency(TfidfRag):
    """Adil taban: benzerlik + son N olay.

    Duz TF-IDF uzun kosularda tool_update'i kaciriyor -- tek bir olay,
    yuzlerce cagrinin arasinda benzerlik siralamasinda one cikamiyor.
    Gercek RAG sistemleri recency de kullanir; bu kol onu temsil eder.
    Yoksa taban strawman olur.
    """
    name = "TF-IDF + recency (hibrit)"

    def __init__(self, k=12, recent=8):
        super().__init__(k=k)
        self.recent = recent

    def read(self, tool):
        top = super().read(tool)
        tail = [ev for ev, _ in self.docs][-self.recent:]
        seen, out = set(), []
        for ev in tail + top:                 # yeniler once
            key = (ev["step"], ev["type"])
            if key not in seen:
                seen.add(key)
                out.append(ev)
        return out
