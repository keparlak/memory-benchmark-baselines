"""Deney 5 — schema-bump korpusu.

Bir agent kosusunun ortasinda tool semasi degisir. Korpus, o kosuyu ve
degisimi uretir; puanlama tamamen deterministiktir (JSON Schema).

Tasarim: DENEY5-tasarim-schema-bump.md
Kaynak fikir: @kartikb753, 22 Eyl 2026.
"""
import json
import random

# --- sema tanimlari -------------------------------------------------------

BASE_TOOLS = {
    "create_ticket": {
        "title": "str", "body": "str", "priority": "str",
    },
    "assign_owner": {
        "ticket_id": "str", "owner": "str",
    },
    "set_due_date": {
        "ticket_id": "str", "due": "str",
    },
    "add_label": {
        "ticket_id": "str", "label": "str",
    },
    "close_ticket": {
        "ticket_id": "str", "reason": "str",
    },
}

# Her mutasyon farkli bir hata imzasi uretir. (tip, tool, ayrinti)
MUTATIONS = [
    ("rename",        "create_ticket", ("title", "subject")),
    ("rename",        "assign_owner",  ("owner", "assignee")),
    ("split",         "create_ticket", ("body", ("description", "summary"))),
    ("add_required",  "create_ticket", ("team", "str")),
    ("add_required",  "set_due_date",  ("timezone", "str")),
    ("type_change",   "create_ticket", ("priority", "severity", "int")),
    ("type_change",   "set_due_date",  ("due", "due_ts", "int")),
    ("remove",        "add_label",     ("label",)),
    ("remove",        "close_ticket",  ("reason",)),
]


def apply_mutation(schema, mut):
    """v1 semasindan v2 semasi uret. Saf fonksiyon, girdiyi bozmaz."""
    kind, tool, detail = mut
    s = {t: dict(fields) for t, fields in schema.items()}
    f = s[tool]
    if kind == "rename":
        old, new = detail
        f[new] = f.pop(old)
    elif kind == "split":
        old, (a, b) = detail
        f.pop(old)
        f[a] = "str"
        f[b] = "str"
    elif kind == "add_required":
        name, typ = detail
        f[name] = typ
    elif kind == "type_change":
        old, new, typ = detail
        f.pop(old)
        f[new] = typ
    elif kind == "remove":
        (old,) = detail
        f.pop(old)
    else:
        raise ValueError(f"bilinmeyen mutasyon: {kind}")
    return s


def validate_call(call, schema):
    """Bir tool cagrisi semaya uyuyor mu? Yorum payi yok.

    Doner: (gecerli_mi, hata_listesi)
    """
    tool = call.get("tool")
    if tool not in schema:
        return False, [f"bilinmeyen tool: {tool}"]
    spec = schema[tool]
    args = call.get("args", {})
    errs = []
    for name, typ in spec.items():
        if name not in args:
            errs.append(f"eksik alan: {name}")
            continue
        v = args[name]
        if typ == "int" and not isinstance(v, int):
            errs.append(f"tip hatasi: {name} int bekleniyordu, {type(v).__name__} geldi")
        elif typ == "str" and not isinstance(v, str):
            errs.append(f"tip hatasi: {name} str bekleniyordu, {type(v).__name__} geldi")
    for name in args:
        if name not in spec:
            errs.append(f"fazla alan: {name}")
    return (not errs), errs


# --- korpus uretimi -------------------------------------------------------

_WORDS = ["login", "billing", "export", "sync", "webhook", "latency",
          "timeout", "upload", "search", "refund"]


def _value_for(field, typ, rng):
    if typ == "int":
        return rng.randint(1, 5)
    return f"{rng.choice(_WORDS)}-{rng.randint(100, 999)}"


def make_call(tool, schema, rng):
    """Verilen semaya gore gecerli bir cagri uret."""
    return {"tool": tool, "args": {n: _value_for(n, t, rng)
                                   for n, t in schema[tool].items()}}


def make_episode(ep_id, rng, n_steps=12, mut_index=None):
    """Tek bir kosu: v1 ile baslar, k* adiminda v2'ye gecer.

    Donen sozlukte dogru cevaplar da bulunur; puanlayici bunlari kullanmaz,
    sadece raporlama icin tutulur.
    """
    v1 = {t: dict(f) for t, f in BASE_TOOLS.items()}
    # rng her durumda tuketilir ki mut_index verilmeyen kosularin (Deney 5)
    # rastgele dizisi degismesin. mut_index: Deney 6'da mutasyon tipini
    # uzunluklar arasinda sabit tutmak icin -- yoksa uzunluk etkisi mutasyon
    # etkisiyle karisir (Deney 5'te remove 0/4, type_change 2/2 idi).
    mut = MUTATIONS[rng.randrange(len(MUTATIONS))]
    if mut_index is not None:
        mut = MUTATIONS[mut_index]
    v2 = apply_mutation(v1, mut)
    kind, mut_tool, _ = mut

    k_star = rng.randint(n_steps // 3, 2 * n_steps // 3)

    events = []
    for i in range(n_steps):
        if i == k_star:
            events.append({
                "step": i,
                "type": "tool_update",
                "tool": mut_tool,
                "mutation": kind,
                "schema": {mut_tool: v2[mut_tool]},
                "text": _update_text(mut_tool, mut, v2),
            })
        schema = v1 if i < k_star else v2
        tool = mut_tool if rng.random() < 0.5 else rng.choice(list(BASE_TOOLS))
        events.append({
            "step": i,
            "type": "call",
            "call": make_call(tool, schema, rng),
            "schema_version": "v1" if i < k_star else "v2",
        })

    # Degerlendirme sorulari: k* sonrasi, mutasyona ugrayan tool icin.
    queries = [{"tool": mut_tool, "intent": f"{mut_tool} cagir"}
               for _ in range(3)]

    return {
        "id": ep_id,
        "v1": v1,
        "v2": v2,
        "mutation": {"kind": kind, "tool": mut_tool},
        "k_star": k_star,
        "events": events,
        "queries": queries,
    }


def _update_text(tool, mut, v2):
    """Sema degisimini duz metinle bildir - gercekte changelog boyle gelir."""
    kind, _, detail = mut
    fields = ", ".join(f"{n}: {t}" for n, t in v2[tool].items())
    if kind == "rename":
        old, new = detail
        head = f"{tool}: '{old}' alani '{new}' olarak yeniden adlandirildi."
    elif kind == "split":
        old, (a, b) = detail
        head = f"{tool}: '{old}' alani '{a}' ve '{b}' olarak ikiye ayrildi."
    elif kind == "add_required":
        name, _typ = detail
        head = f"{tool}: '{name}' artik zorunlu bir alan."
    elif kind == "type_change":
        old, new, typ = detail
        head = f"{tool}: '{old}' kaldirildi, yerine '{new}' ({typ}) geldi."
    else:
        (old,) = detail
        head = f"{tool}: '{old}' alani kaldirildi."
    return f"{head} Guncel imza -> {tool}({fields})"


def build_corpus(n=200, seed=20260922):
    rng = random.Random(seed)
    return [make_episode(i, rng) for i in range(n)]


if __name__ == "__main__":
    corpus = build_corpus()
    with open("deney5_korpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    # kendi kendini dogrula: uretilen her cagri kendi surumune uymali
    bad = 0
    for ep in corpus:
        for ev in ep["events"]:
            if ev["type"] != "call":
                continue
            schema = ep["v1"] if ev["schema_version"] == "v1" else ep["v2"]
            ok, errs = validate_call(ev["call"], schema)
            if not ok:
                bad += 1
    dist = {}
    for ep in corpus:
        k = ep["mutation"]["kind"]
        dist[k] = dist.get(k, 0) + 1

    print(f"kosu: {len(corpus)}")
    print(f"tutarsiz uretilmis cagri: {bad}  (0 olmali)")
    print("mutasyon dagilimi:", dict(sorted(dist.items())))

    # v1 cagrisi v2'ye uymamali - test anlamli mi kontrolu
    leak = 0
    for ep in corpus:
        pre = [e for e in ep["events"]
               if e["type"] == "call"
               and e["schema_version"] == "v1"
               and e["call"]["tool"] == ep["mutation"]["tool"]]
        for e in pre:
            ok, _ = validate_call(e["call"], ep["v2"])
            if ok:
                leak += 1
    print(f"v1 cagrisi v2'ye de uyuyor: {leak}  (0 olmali - yoksa test korlesir)")
