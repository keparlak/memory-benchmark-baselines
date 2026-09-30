"""Deney 6 analizi: uzunluga gore Mem0 retrieval, iki sorgu, kontroller.

Kurallar (Deney 5 derslerinden):
  - servis hatasi (None) paydaya girmez
  - m6 (due -> due_ts) icin "eski kanit" sayilmaz: 'due' ayni zamanda
    Ingilizce kelime ("due date"), kelime sinirli eslesme de ayirt edemiyor
  - yazma hatasi olan kosular ayrica isaretlenir
"""
import glob
import json
import sys
from collections import defaultdict

rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob("results/deney6/*.json"))]
if not rows:
    sys.exit("veri yok")
STALE_UNRELIABLE = {6}


def rate(xs):
    xs = [x for x in xs if x is not None]
    return (100.0 * sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def fmt(v):
    p, n = v
    return f"{p:5.0f}%/{n}" if n else "   -  "


by = defaultdict(list)
for r in rows:
    by[r["length"]].append(r)
Ls = sorted(by)

print(f"kosu: {len(rows)}   uzunluklar: {Ls}\n")
print("=== ajan dogrulugu (validity@v2) ===")
print(f"{'uzunluk':>7} {'depo':>5} {'upd kaydi var':>13} {'mem0 schemaQ':>13} {'mem0 taskQ':>11} {'hafizasiz':>10} {'supersess':>10}")
for L in Ls:
    g = by[L]
    upd = sum(1 for r in g if r["store_has_update"])
    print(f"{L:>7} {sum(r['store_size'] for r in g)/len(g):>5.0f} {upd:>7}/{len(g):<5} "
          f"{fmt(rate([r['mem0_ok'] for r in g])):>13} {fmt(rate([r['mem0_ok_taskq'] for r in g])):>11} "
          f"{fmt(rate([r['nomem_ok'] for r in g])):>10} {fmt(rate([r['supersession_ok'] for r in g])):>10}")

for q in ("schema_q", "task_q"):
    print(f"\n=== siralama ({q}) — ajana giden ilk 20 ===")
    print(f"{'uzunluk':>7} {'upd top20de':>12} {'upd sirasi (medyan)':>20} {'yeni/20':>8} {'eski/20*':>9}")
    for L in Ls:
        g = by[L]
        has = [r for r in g if r["store_has_update"]]
        in_top = sum(1 for r in has if r[q]["rank_update"] is not None and r[q]["rank_update"] < 20)
        ranks = sorted(r[q]["rank_update"] for r in has if r[q]["rank_update"] is not None)
        medr = ranks[len(ranks) // 2] if ranks else None
        fr = sum(r[q]["top20_fresh"] for r in g) / len(g)
        st_rows = [r for r in g if r["v1_only"] and r["mut_index"] not in STALE_UNRELIABLE]
        st = sum(r[q]["top20_stale"] for r in st_rows) / len(st_rows) if st_rows else float("nan")
        print(f"{L:>7} {in_top:>6}/{len(has):<5} {str(medr):>20} {fr:>8.1f} {st:>9.1f}")
print("  * eski kanit: m6 haric (due/'due date' cakismasi)")

print("\n=== basari ile 'degisiklik kaydi var mi' iliskisi (mem0, schemaQ) ===")
tab = defaultdict(lambda: [0, 0])
for r in rows:
    if r["mem0_ok"] is None:
        continue
    k = "upd kaydi VAR" if r["store_has_update"] else "upd kaydi YOK"
    tab[k][0] += r["mem0_ok"]
    tab[k][1] += 1
for k, (ok, n) in sorted(tab.items()):
    print(f"  {k}: {ok}/{n} basarili")

slots = defaultdict(int)
for r in rows:
    for k, v in r["slots"].items():
        slots[k] += v
bad = [f"L{r['length']}m{r['mut_index']}" for r in rows if r["write_errors"]]
print(f"\nslot kararlari: {dict(slots)}   yazma hatali kosular: {bad or 'yok'}")

print("\n=== mutasyon x uzunluk (mem0 schemaQ / taskQ) ===")
muts = sorted({(r["mut_index"], r["mutation"]) for r in rows})
print(f"{'':18}" + "".join(f"{L:>9}" for L in Ls))
sym = {True: "ok", False: "--", None: "ER"}
for mi, m in muts:
    cells = []
    for L in Ls:
        rr = [r for r in by[L] if r["mut_index"] == mi]
        cells.append(f"{(sym[rr[0]['mem0_ok']] + '/' + sym[rr[0]['mem0_ok_taskq']]) if rr else '':>9}")
    print(f"{m + ' m' + str(mi):18}" + "".join(cells))
