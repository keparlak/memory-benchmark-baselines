"""Teshis: neden stored (get_all) < retrieved (search)?

Saklanmayan bir sey getirilemez; bu tutarsizlik olcumun bir yerinin bozuk
oldugunu gosteriyor. Deney 6 (uzunluk / "retrieval bogulur mu") tam olarak
get_all ile search'u karsilastiracagi icin once bunun cozulmesi gerek.

Tek kosu yazar, sonra iki yolun dondurdugu kayitlari ID duzeyinde karsilastirir.
"""
import inspect
import random
import warnings

warnings.filterwarnings("ignore")

import deney5_mem0 as M
from deney5_kollar import SIG
from deney5_korpus import make_episode


def sigs(text):
    out = []
    for m in SIG.finditer(text or ""):
        fields = sorted(p.split(":")[0].strip() for p in m.group(2).split(",") if ":" in p)
        out.append((m.group(1), fields))
    return out


def main():
    # Deney 5 parca 0'daki ep0 (type_change, stored=False retr=True) ile ayni tohum
    import deney5_asama2 as A
    A.N_EPISODES = 1
    ep = A.build_corpus()[0]
    tool = ep["mutation"]["tool"]
    v2 = sorted(ep["v2"][tool])
    print(f"mutasyon={ep['mutation']}  v2={v2}  olay={len(ep['events'])}", flush=True)

    arm = M.Mem0Arm(batch=10)
    for ev in ep["events"]:
        arm.write(ev)
    arm._flush()
    m, uid = arm.m, arm.user_id

    print("\nget_all imzasi:", inspect.signature(m.get_all), flush=True)
    print("search  imzasi:", inspect.signature(m.search), flush=True)

    def items(r):
        return r.get("results", r) if isinstance(r, dict) else (r or [])

    ga = items(m.get_all(filters={"user_id": uid}, limit=10000))
    ga_default = items(m.get_all(filters={"user_id": uid}))
    sr = items(m.search(query=f"{tool} current schema arguments",
                        filters={"user_id": uid}, limit=12))

    ga_ids = {x.get("id") for x in ga}
    print(f"\nget_all(limit=10000): {len(ga)} kayit   get_all(varsayilan): {len(ga_default)} kayit")
    print(f"search(limit=12)    : {len(sr)} kayit")
    missing = [x for x in sr if x.get("id") not in ga_ids]
    print(f"search'te olup get_all'da OLMAYAN: {len(missing)}", flush=True)

    print("\n--- search sonuclari (imza iceren) ---")
    for i, x in enumerate(sr):
        s = sigs(x.get("memory"))
        if s:
            tag = "GET_ALL'DA YOK" if x.get("id") not in ga_ids else "get_all'da var"
            print(f" [{i}] {tag}  imzalar={s}\n      {x.get('memory','')[:160]}")

    print("\n--- get_all'da imza iceren kayitlar ---")
    for x in ga:
        s = sigs(x.get("memory"))
        if s:
            print(f"  imzalar={s}\n      {x.get('memory','')[:160]}")

    print("\nstored_v2  :", arm.stored_v2(tool, set(v2)))
    ctx = arm.read(tool)
    print("retrieved  :", any(e.get("type") == "tool_update" and e.get("tool") == tool
                              and sorted(e["schema"][tool]) == v2 for e in ctx))
    arm.close()
    M.Mem0Arm.shutdown()


if __name__ == "__main__":
    main()
