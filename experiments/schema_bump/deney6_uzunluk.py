"""Deney 6 -- depo buyudukce retrieval bogulur mu?

Kaynak soru (@kartikb753, 23 Eyl 2026):
  "An append-only memory can never say this is gone, only this is newer.
   The question is whether anything downstream ever garbage collects, or if
   the store just grows until retrieval drowns."

Neden adil bir soru: mem0 v2.0.0 (2026-04-14) tasarim geregi ADD-only.
Belgesi: "When information changes, the new fact is stored alongside the old
one. Retrieval handles ranking: the most relevant, current information
surfaces first." (docs.mem0.ai/migration/oss-v2-to-v3)
Ama acik kaynak siralama fonksiyonu (mem0/utils/scoring.py score_and_rank)
yalnizca anlamsal + BM25 + varlik puani aliyor; zaman terimi yok.
Bu deney belgedeki iddiayi olcer: depo buyudukce guncel bilgi hala one cikiyor mu?

Tasarim:
  - uzunluk: 20 / 60 / 120 / 200 adim; mutasyon tipi sabit (uzunluk tek degisken)
  - mutasyonlar: v2'ye ozgu alani olanlar (rename x2, add_required x2, type_change x2)
  - her kosu kendi JSON'una yazilir; var olan atlanir; zaman butcesi dolunca durur
  - kontroller her uzunlukta: tum gecmis (hafizasiz) + deterministik supersession

Kullanim: python deney6_uzunluk.py [butce_saniye]
"""
import json
import os
import random
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import deney5_asama2 as A
import deney5_llm as L
import deney5_mem0 as M
from deney5_kollar import NaiveAppend, SupersessionPolicy, SIG
from deney5_korpus import make_episode, validate_call

OUT = "results/deney6"
LENGTHS = [20, 60, 120, 200]
MUTS = [0, 1, 3, 4, 5, 6]      # rename, rename, add_required, add_required, type_change, type_change
SEED = 20260930
RANK_TOP_K = 100              # sira analizi icin genis arama
SEC_PER_STEP = 1.1            # olculen: L=120 ~120s
CHANGE_WORDS = ("signature", "renamed", "replaced", "removed", "required",
                "updated", "changed", "instead", "no longer", "split", "now")
# Sentetik korpusta cagri kayitlarinin degerleri hep "kelime-ddd" (login-401).
# Degisikligi anlatan kayitta boyle deger olmaz -> saglam ayirici.
VALUE_TOKEN = re.compile(r"\b[a-z]+-\d{3}\b")


def agent_ok(tool, ctx, ep):
    try:
        call = A.ask(tool, ctx, A.TASKS[ep["id"] % len(A.TASKS)])
    except L.LLMError:
        return None                      # servis hatasi: paydaya girmez
    if call is None or call.get("tool") != tool:
        return False
    ok, _ = validate_call(call, ep["v2"])
    return bool(ok)


def has_field(text, f):
    """Kelime sinirli eslesme. Ilk surum alt-dize kullaniyordu ve 'due',
    'due_ts'nin icinde gectigi icin her yeni kayit ayni zamanda 'eski'
    sayiliyordu (top20 fresh/stale = 15/20 gibi imkansiz sonuclar)."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}(?![A-Za-z0-9_])", text) is not None


def classify(text, tool, v2_only, v1_only):
    t = text or ""
    fresh = bool(v2_only) and all(has_field(t, f) for f in v2_only)
    stale = bool(v1_only) and any(has_field(t, f) for f in v1_only)
    is_call = VALUE_TOKEN.search(t) is not None
    low = t.lower()
    is_update = (not is_call) and bool(v2_only) and any(has_field(t, f) for f in v2_only) \
        and any(w in low for w in CHANGE_WORDS)
    return fresh, stale, is_update


def rank_stats(texts, tool, v2_only, v1_only):
    cls = [classify(t, tool, v2_only, v1_only) for t in texts]
    top = cls[:M.SEARCH_TOP_K]
    return {
        "rank_fresh": next((i for i, c in enumerate(cls) if c[0]), None),
        "rank_update": next((i for i, c in enumerate(cls) if c[2]), None),
        "top20_fresh": sum(1 for c in top if c[0]),
        "top20_stale": sum(1 for c in top if c[1]),
    }


def run_one(length, mut_index):
    path = f"{OUT}/L{length:03d}_m{mut_index}.json"
    if os.path.exists(path):
        return None
    rng = random.Random(SEED * 100 + mut_index)
    ep = make_episode(mut_index, rng, n_steps=length, mut_index=mut_index)
    tool = ep["mutation"]["tool"]
    v2f = set(ep["v2"][tool])
    v2_only = sorted(v2f - set(ep["v1"][tool]))
    v1_only = sorted(set(ep["v1"][tool]) - v2f)
    t0 = time.time()

    arm = M.Mem0Arm(batch=10)
    for ev in ep["events"]:
        arm.write(ev)
    arm._flush()

    store = arm.all_texts()
    task = A.TASKS[ep["id"] % len(A.TASKS)]
    # iki sorgu: harness'in "schema" sorgusu Mem0'in lehine olabilir
    # (degisikligi anlatan kayda dogal olarak yakin). Gercek ajan gorevle arar.
    q_schema = f"{tool} current schema arguments"
    q_task = f"call {tool} to {task}"
    ranked = arm.search_texts(q_schema, RANK_TOP_K)
    ranked_task = arm.search_texts(q_task, RANK_TOP_K)
    rs = rank_stats(ranked, tool, v2_only, v1_only)
    rt = rank_stats(ranked_task, tool, v2_only, v1_only)

    ctx = arm.read(tool)                     # ajana giden: top_k=20, schema sorgusu
    mem_ok = agent_ok(tool, ctx, ep)
    # ayni ajan, gorev sorgusuyla getirilen baglam
    ctx_task = arm._as_events(ranked_task[:M.SEARCH_TOP_K])
    mem_ok_task = agent_ok(tool, ctx_task, ep)
    slots = {}
    for s in arm.slot_log:
        e = s.get("event") or "NONE"
        slots[e] = slots.get(e, 0) + 1
    errors = arm.errors
    arm.close()

    # kontroller: ayni kosu, ayni ajan
    ctrl = {}
    for name, mk in (("nomem", NaiveAppend), ("supersession", SupersessionPolicy)):
        mem = mk()
        for ev in ep["events"]:
            mem.write(ev)
        ctrl[name] = agent_ok(tool, mem.read(tool), ep)

    row = {
        "length": length, "events": len(ep["events"]), "mut_index": mut_index,
        "mutation": ep["mutation"]["kind"], "tool": tool,
        "v2_only": v2_only, "v1_only": v1_only,
        "store_size": len(store),
        "stored_fresh": any(all(f in s for f in v2_only) for s in store) if v2_only else None,
        "schema_q": rs, "task_q": rt,
        "mem0_ok": mem_ok, "mem0_ok_taskq": mem_ok_task,
        "nomem_ok": ctrl["nomem"], "supersession_ok": ctrl["supersession"],
        "slots": slots, "write_errors": errors, "seconds": round(time.time() - t0),
        # ham metinler: "degisiklik hic kaydedilmemis mi" sonradan dogrulanabilsin
        "store_has_update": any(classify(s, tool, v2_only, v1_only)[2] for s in store),
        "store_texts": store,
        "ranked_texts": ranked[:M.SEARCH_TOP_K],
        "ranked_texts_task": ranked_task[:M.SEARCH_TOP_K],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    return row


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 480
    probe = M.Mem0Arm()
    base = str(probe.m.llm.client.base_url)
    probe.close()
    if "nvidia" not in base:
        print(f"DURDURULDU: yanlis uc {base}", flush=True)
        M.Mem0Arm.shutdown()
        return 2
    t0 = time.time()
    done = 0
    for length in LENGTHS:
        for mi in MUTS:
            if os.path.exists(f"{OUT}/L{length:03d}_m{mi}.json"):
                continue
            # buyuk kosular butceyi asmasin: kaba sure tahmini adim basina ~1.9s
            if time.time() - t0 + length * SEC_PER_STEP > budget and done > 0:
                print(f"butce doldu, {done} kosu yazildi", flush=True)
                M.Mem0Arm.shutdown()
                return 0
            r = run_one(length, mi)
            done += 1
            s, t = r["schema_q"], r["task_q"]
            print(f"L={length:3} {r['mutation']:12} store={r['store_size']:3} upd_in_store={r['store_has_update']} | "
                  f"schemaQ upd@{s['rank_update']} f/s={s['top20_fresh']}/{s['top20_stale']} ok={r['mem0_ok']} | "
                  f"taskQ upd@{t['rank_update']} f/s={t['top20_fresh']}/{t['top20_stale']} ok={r['mem0_ok_taskq']} | "
                  f"nomem={r['nomem_ok']} sup={r['supersession_ok']} slots={r['slots']} "
                  f"err={r['write_errors']} {r['seconds']}s", flush=True)
    M.Mem0Arm.shutdown()
    print(f"TAMAM, bu turda {done} kosu", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
