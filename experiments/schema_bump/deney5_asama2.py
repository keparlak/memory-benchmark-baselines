"""Deney 5 — Asama 2: gercek modelle kollar.

Asama 1'de ajan deterministikti: baglamda ne varsa onu kullaniyordu.
Burada ajan gercek bir LLM. Soru degismedi:

    sema kosunun ortasinda degistiginde, hafiza katmani ajana
    guncel semayi verebiliyor mu -- ve veremiyorsa kayip nerede?

Kollar (hepsi ayni model, ayni prompt):
    A  no-memory      : tum kosu baglama konur. Tavan.
    C  tfidf+recency  : benim tabanim
    D  supersession   : deterministik politika
    E  lossy          : write path bozuk referans egrisi
    B  mem0           : gercek urun (deney5_mem0.py ile eklenir)

Olculen sey ajanin zekasi degil, hafizanin ona ne verdigidir; bu yuzden
prompt ve model tum kollarda ayni tutulur.
"""
import json
import random
import sys
import time
from collections import Counter

from deney5_korpus import make_episode, validate_call
from deney5_kollar import (NaiveAppend, TfidfRecency, SupersessionPolicy,
                           LossyConsolidate)
import deney5_llm as L

N_EPISODES = 40
N_STEPS = 60
SEED = 20260922

SYS = ("You are an agent that calls tools. The tool schema can change "
       "mid-session; always use the CURRENT schema. "
       "Output exactly one JSON object of the form "
       "{\"tool\": \"<name>\", \"args\": {...}} and nothing else.")


def render(events):
    """Baglami duz metne cevir. Tum kollarda ayni bicim.

    "note": Mem0 ham olay dondurmuyor, kendi cumlesiyle yeniden yazilmis
    metin donduruyor. Onu oldugu gibi gecireriz -- urunun ciktisini
    kendi bicimime zorlamak, olctugum seyi degistirmek olurdu.
    """
    out = []
    for ev in sorted(events, key=lambda e: e.get("step", 0)):
        t = ev.get("type")
        if t == "tool_update":
            out.append(f"[tool_update] {ev['text']}")
        elif t == "call":
            c = ev["call"]
            args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
            out.append(f"  {c['tool']}({args})")
        else:
            out.append(f"  {ev.get('text', '')}")
    return "\n".join(out)


def ask(tool, context_events, task):
    """Ajan cagrisi.

    Reasoning kapali (deney5_llm varsayilani): 8 kosuda thinking acik ve
    kapali ayni sonucu verdi (8/8), kapali 2.6x hizli. Acik birakmak sadece
    bos content riskini buyutuyordu.
    """
    ctx = render(context_events) or "(no history)"
    user = f"Log:\n{ctx}\n\nTask: call {tool} to {task}."
    return L.parse_call(L.chat([{"role": "system", "content": SYS},
                                {"role": "user", "content": user}],
                               max_tokens=800))


def build_corpus():
    rng = random.Random(SEED)
    return [make_episode(i, rng, n_steps=N_STEPS) for i in range(N_EPISODES)]


TASKS = ["record that webhook retries are exhausted",
         "record that the nightly sync failed",
         "record that search latency spiked"]


def run_arm(corpus, make_mem, name, log):
    st = {"valid": 0, "total": 0, "stored_v2": 0, "retrieved_v2": 0,
          "no_call": 0, "llm_fail": 0, "attribution": Counter(),
          "by_mutation": {}, "loose_n": 0, "loose_hit": 0,
          "slots": Counter()}
    for ep in corpus:
        mem = make_mem()
        for ev in ep["events"]:
            mem.write(ev)

        tool = ep["mutation"]["tool"]
        kind = ep["mutation"]["kind"]
        v2f = set(ep["v2"][tool])

        v2_only = set(ep["v2"][tool]) - set(ep["v1"][tool])
        v1_only = set(ep["v1"][tool]) - set(ep["v2"][tool])

        stored = mem.stored_v2(tool, v2f)                       # KATI olcut
        loose = mem.stored_v2_loose(tool, v2_only, v1_only)     # GEVSEK olcut
        ctx = mem.read(tool)
        retrieved = any(e.get("type") == "tool_update" and e.get("tool") == tool
                        and set(e["schema"][tool]) == v2f for e in ctx)

        if loose is not None:
            st["loose_n"] += 1
            st["loose_hit"] += int(loose)
        for s_ in (getattr(mem, "slot_log", None) or []):
            st["slots"][s_.get("event") or "NONE"] += 1

        task = TASKS[ep["id"] % len(TASKS)]
        ok = False
        try:
            call = ask(tool, ctx, task)
            if call is None:
                st["no_call"] += 1
            elif call.get("tool") != tool:
                pass                       # yanlis tool -> gecersiz
            else:
                ok, _ = validate_call(call, ep["v2"])
        except L.LLMError as e:
            st["llm_fail"] += 1
            log.write(f"{name}\tep{ep['id']}\tLLM_FAIL\t{e}\n")

        st["total"] += 1
        st["valid"] += int(ok)
        st["stored_v2"] += int(stored)
        st["retrieved_v2"] += int(retrieved)
        b = st["by_mutation"].setdefault(kind, [0, 0])
        b[0] += int(ok); b[1] += 1
        st["attribution"][("v2 saklandi" if stored else "v2 SAKLANMADI",
                           "v2 getirildi" if retrieved else "v2 getirilmedi")] += 1
        log.flush()
        close = getattr(mem, "close", None)
        if close:
            close()                     # Mem0: bu kosunun kayitlarini temizle
    return st


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def _mem0():
    from deney5_mem0 import Mem0Arm
    return Mem0Arm(batch=10)


ARMS = [
    ("A no-memory (tavan)", lambda: NaiveAppend()),
    ("B mem0", _mem0),
    ("C tfidf+recency", lambda: TfidfRecency(k=12, recent=6)),
    ("D supersession", lambda: SupersessionPolicy()),
    ("E lossy", lambda: LossyConsolidate()),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    corpus = build_corpus()
    print(f"korpus: {N_EPISODES} kosu x ~{len(corpus[0]['events'])} olay, "
          f"model {L.MODEL}\n")

    res = {}
    t0 = time.time()
    with open("deney5_asama2.log", "a", encoding="utf-8") as log:
        for name, mk in ARMS:
            if only and not any(o in name for o in only):
                continue
            t = time.time()
            st = run_arm(corpus, mk, name, log)
            res[name] = st
            loose = pct(st['loose_hit'], st['loose_n']) if st['loose_n'] else float('nan')
            slots = dict(st["slots"]) if st["slots"] else {}
            print(f"{name:22} validity={pct(st['valid'],st['total']):5.1f}%  "
                  f"stored(kati)={pct(st['stored_v2'],st['total']):5.1f}%  "
                  f"stored(gevsek)={loose:5.1f}%  "
                  f"retr={pct(st['retrieved_v2'],st['total']):5.1f}%  "
                  f"(no_call={st['no_call']} llm_fail={st['llm_fail']}) "
                  f"{time.time()-t:.0f}s")
            if slots:
                print(f"{'':22} slot kararlari: {slots}")

    try:
        from deney5_mem0 import Mem0Arm
        Mem0Arm.shutdown()
    except Exception:
        pass

    print(f"\ntoplam {time.time()-t0:.0f}s  llm={L.STATS}")
    with open("deney5_asama2_sonuc.json", "w", encoding="utf-8") as f:
        def _ser(k, v):
            if k == "attribution":
                return dict((f"{a}|{b}", c) for (a, b), c in v.items())
            if k == "slots":
                return dict(v)
            return v
        json.dump({n: {k: _ser(k, v) for k, v in s.items()}
                   for n, s in res.items()}, f, ensure_ascii=False, indent=2)
    print("-> deney5_asama2_sonuc.json")
