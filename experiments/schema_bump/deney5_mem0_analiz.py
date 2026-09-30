"""Deney 5 -- Mem0 kolunun birlestirilmis analizi (results/deney5_mem0/*.json).

Not: bu koşularda get_all mem0 2.1.0'da sessizce yok sayilan `limit` ile
cagrildi (dogrusu `top_k`), dolayisiyla `stored`/`loose` alanlari deponun
yalnizca ~20 kaydina bakiyor ve GUVENILMEZ. `ok` (ajanin cagrisi) ve `slots`
(add() donusu) bundan etkilenmiyor.
"""
import glob
import json
from collections import Counter, defaultdict

rows = []
for p in sorted(glob.glob("results/deney5_mem0/*.json")):
    rows += json.load(open(p, encoding="utf-8"))

n = len(rows)
ok = sum(r["ok"] for r in rows)
slots = Counter()
for r in rows:
    slots.update(r["slots"])

print(f"Mem0 (Deney 5): {n} kosu, validity@v2 = {ok}/{n} = {100 * ok / n:.1f}%")
print(f"slot kararlari: {dict(slots)}  (yazma hatasi: {sum(r['errors'] for r in rows)})")
by = defaultdict(lambda: [0, 0])
for r in rows:
    by[r["mutation"]][0] += r["ok"]
    by[r["mutation"]][1] += 1
print("mutasyon tipine gore:")
for k in sorted(by):
    print(f"  {k:13} {by[k][0]}/{by[k][1]}")
