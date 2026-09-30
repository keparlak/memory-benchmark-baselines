"""Mem0 kolunu PARCALI kos: 3'er kosuluk kisa surecler.

Neden: 15 kosuluk tek surec bellekte ~10 dk kaliyor ve sistem bellek
baskisi altinda arka plan islerini kesiyor (4 denemede 4 kesilme).
Kisa surec bellekte ~2 dk kalir; kesilme riski dusuk, kesilse bile
sadece o parca kaybolur, oncekiler diskte durur.

Her parca kendi JSON'unu yazar; birlestirme ayri yapilir.

Kullanim:
    python deney5_mem0_parca.py <baslangic> <adet>
    ornek: python deney5_mem0_parca.py 0 3    -> ep 0,1,2
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import deney5_asama2 as A
import deney5_llm as L
import deney5_mem0 as M
from deney5_korpus import validate_call

OUT = "results/deney5_mem0"


def main():
    start = int(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    A.N_EPISODES = start + count
    corpus = A.build_corpus()[start:start + count]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/ep{start:03d}_{count}.json"
    if os.path.exists(path):
        print(f"zaten var, atlandi: {path}", flush=True)
        return 0

    probe = M.Mem0Arm()
    base = str(probe.m.llm.client.base_url)
    probe.close()
    if "nvidia" not in base:
        print(f"DURDURULDU: yanlis uc {base}", flush=True)
        M.Mem0Arm.shutdown()
        return 2
    print(f"ep{start}..{start+count-1}  uc={base}", flush=True)

    rows = []
    t0 = time.time()
    for ep in corpus:
        mem = M.Mem0Arm(batch=10)
        for ev in ep["events"]:
            mem.write(ev)
        tool = ep["mutation"]["tool"]
        v2f = set(ep["v2"][tool])
        v2_only = v2f - set(ep["v1"][tool])

        stored = mem.stored_v2(tool, v2f)
        loose = mem.stored_v2_loose(tool, v2_only, None)
        ctx = mem.read(tool)
        retrieved = any(e.get("type") == "tool_update" and e.get("tool") == tool
                        and set(e["schema"][tool]) == v2f for e in ctx)
        ok, no_call, fail = False, 0, 0
        try:
            call = A.ask(tool, ctx, A.TASKS[ep["id"] % len(A.TASKS)])
            if call is None:
                no_call = 1
            elif call.get("tool") == tool:
                ok, _ = validate_call(call, ep["v2"])
        except L.LLMError:
            fail = 1
        slots = {}
        for s in mem.slot_log:
            e = s.get("event") or "NONE"
            slots[e] = slots.get(e, 0) + 1
        rows.append({"ep": ep["id"], "mutation": ep["mutation"]["kind"],
                     "ok": bool(ok), "stored": bool(stored),
                     "loose": (None if loose is None else bool(loose)),
                     "retrieved": bool(retrieved), "no_call": no_call,
                     "fail": fail, "slots": slots,
                     "add_calls": mem.add_calls, "errors": mem.errors})
        print(f"  ep{ep['id']:3} {ep['mutation']['kind']:13} ok={str(ok):5} "
              f"stored={str(stored):5} loose={str(loose):5} "
              f"retr={str(retrieved):5} slots={slots}", flush=True)
        mem.close()

    M.Mem0Arm.shutdown()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
