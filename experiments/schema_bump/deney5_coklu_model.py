"""Deney 5 — coklu model: bulgu modele mi ait, katmana mi?

Iki eksen, ikisi de NVIDIA NIM icinde (gunluk limit yok):

  GUC   nemotron-3-super-120b  ->  nemotron-3-ultra-550b   (ayni aile, 4.5x)
  AILE  nemotron              ->  deepseek-v4.1-flash      (farkli mimari)

Kollar Mem0'siz: A (tavan), C (tfidf), D (supersession), E (lossy).
Mem0 bu makinede bellek yuzunden kosturulamiyor; ayrica onun katkisi
slot kararlarindan geldi, ajan modelinden degil.

Sorulan sey: ajan modeli degisince TESHIS degisiyor mu?
  - D hep 100'e yakinsa  -> politika modelden bagimsiz calisiyor
  - C hep 0'a yakinsa    -> retrieval basarisizligi modelden bagimsiz
  - A model gucuyle artiyorsa -> tavan modele bagli, bu beklenen

Kullanim: python deney5_coklu_model.py [kosu_sayisi]
"""
import json
import sys
import time

import deney5_asama2 as A
import deney5_llm as L
from deney5_kollar import (NaiveAppend, TfidfRecency, SupersessionPolicy,
                           LossyConsolidate)
from deney5_korpus import validate_call

MODELS = [
    ("nemotron-120b", "nvidia/nemotron-3-super-120b-a12b"),
    ("nemotron-550b", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("deepseek-flash", "deepseek-ai/deepseek-v4.1-flash"),
]

ARMS = [
    ("A no-memory", lambda: NaiveAppend()),
    ("C tfidf+rec", lambda: TfidfRecency(k=12, recent=6)),
    ("D supersess", lambda: SupersessionPolicy()),
    ("E lossy", lambda: LossyConsolidate()),
]


def ask_with(model, tool, ctx_events, task):
    ctx = A.render(ctx_events) or "(no history)"
    user = f"Log:\n{ctx}\n\nTask: call {tool} to {task}."
    return L.parse_call(L.chat(
        [{"role": "system", "content": A.SYS}, {"role": "user", "content": user}],
        model=model, max_tokens=800))


def run(corpus, make_mem, model):
    ok = n = no_call = fail = 0
    for ep in corpus:
        mem = make_mem()
        for ev in ep["events"]:
            mem.write(ev)
        tool = ep["mutation"]["tool"]
        ctx = mem.read(tool)
        task = A.TASKS[ep["id"] % len(A.TASKS)]
        n += 1
        try:
            call = ask_with(model, tool, ctx, task)
            if call is None:
                no_call += 1
            elif call.get("tool") == tool:
                v, _ = validate_call(call, ep["v2"])
                ok += v
        except L.LLMError:
            fail += 1
    return {"ok": ok, "n": n, "no_call": no_call, "fail": fail}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    A.N_EPISODES = n
    corpus = A.build_corpus()
    print(f"korpus: {n} kosu x {len(corpus[0]['events'])} olay\n", flush=True)

    res = {}
    for mlabel, mid in MODELS:
        print(f"--- {mlabel} ({mid}) ---", flush=True)
        for alabel, mk in ARMS:
            t = time.time()
            r = run(corpus, mk, mid)
            pct = 100.0 * r["ok"] / r["n"] if r["n"] else 0.0
            res[f"{mlabel}|{alabel}"] = {**r, "pct": pct}
            print(f"  {alabel:12} {pct:5.1f}%  "
                  f"(no_call={r['no_call']} fail={r['fail']}) "
                  f"{time.time()-t:.0f}s", flush=True)
        print("", flush=True)

    with open("deney5_coklu_model_sonuc.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=== ozet: validity@v2 ===", flush=True)
    print(f"{'kol':14}" + "".join(f"{m:>16}" for m, _ in MODELS), flush=True)
    for alabel, _ in ARMS:
        row = "".join(f"{res[f'{m}|{alabel}']['pct']:15.1f}%" for m, _ in MODELS)
        print(f"{alabel:14}{row}", flush=True)
    print(f"\nllm={L.STATS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
