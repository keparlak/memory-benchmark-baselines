"""
Robustness test: how much of the FactConsolidation score is template knowledge?

Rewrite a share p of the facts into equivalent surface forms the parser does not
know, keeping serial numbers intact. Accuracy then separates "template knowledge"
from "task difficulty".

Result: 89.0 -> 0.0 as p goes 0% -> 100%. All of it is template knowledge.
That makes the FactConsolidation result a benchmark critique, not a method.
"""
import re, random, sys
from factconsolidation_baseline import (FACT_PATS, CUES, parse_facts, build_index,
                       find_entity, answer, subem, cues_in)

# canonical form -> equivalent alternative surface forms
# (same (subject, relation, object) triple, surface form the parser does not know)
PARAPHRASE = {
    'citizen':          ["The country of citizenship of {s} is {o}",
                         "{s} holds citizenship of {o}",
                         "{s}, a national of {o}"],
    'sport':            ["The sport linked to {s} is {o}",
                         "{s} belongs to the sport {o}",
                         "In terms of sport, {s} falls under {o}"],
    'capital':          ["{o} serves as the capital of {s}",
                         "The capital city of {s}: {o}",
                         "{s} has {o} as its capital"],
    'spouse':           ["The spouse of {s} is {o}",
                         "{s} and {o} are married",
                         "{o} is the marital partner of {s}"],
    'died_city':        ["The place of death of {s} is {o}",
                         "{s} passed away in {o}",
                         "Death occurred for {s} in the city {o}"],
    'continent':        ["The continent containing {s} is {o}",
                         "{s} sits within the continent {o}",
                         "Continent of {s}: {o}"],
    'founded_by':       ["The founder of {s} is {o}",
                         "{s} owes its founding to {o}",
                         "{o} established {s}"],
    'author':           ["{o} wrote {s}",
                         "The writer of {s} is {o}",
                         "Authorship of {s} belongs to {o}"],
    'performer':        ["{o} performed {s}",
                         "The performer behind {s} is {o}",
                         "Performance of {s} is credited to {o}"],
    'educated_at':      ["{s} studied at {o}",
                         "The alma mater of {s} is {o}",
                         "Education of {s} took place at {o}"],
    'position':         ["The playing position of {s} is {o}",
                         "{s} occupies the {o} role",
                         "On the field {s} is a {o}"],
    'ceo':              ["{o} leads {s} as chief executive",
                         "The CEO at {s} is {o}",
                         "Executive leadership of {s}: {o}"],
    'hq_city':          ["{s} is headquartered in {o}",
                         "The head office of {s} sits in {o}",
                         "Headquarters city for {s}: {o}"],
    'director':         ["{o} directs {s}",
                         "The head director of {s} is {o}",
                         "Direction of {s} is handled by {o}"],
    'speaks':           ["{s} uses the language {o}",
                         "The spoken language of {s} is {o}",
                         "Language spoken by {s}: {o}"],
    'born_city':        ["The birthplace of {s} is {o}",
                         "{s} was born at {o}",
                         "Birth city for {s}: {o}"],
    'founded_city':     ["{s} began in the city {o}",
                         "The founding city of {s} is {o}",
                         "Origin city of {s}: {o}"],
    'employer':         ["{s} works for {o}",
                         "The employer of {s} is {o}",
                         "Employment of {s} is with {o}"],
    'worked_city':      ["{s} was active in the city {o}",
                         "The working city of {s} is {o}",
                         "Place of work for {s}: {o}"],
    'religion':         ["The religion of {s} is {o}",
                         "{s} follows {o}",
                         "Religious affiliation of {s}: {o}"],
    'created_country':  ["The origin country of {s} is {o}",
                         "{s} originated in {o}",
                         "Country where {s} began: {o}"],
    'created_by':       ["{o} invented {s}",
                         "The originator of {s} is {o}",
                         "Creation of {s} is due to {o}"],
    'official_language':["{s} officially uses {o}",
                         "The state language of {s} is {o}",
                         "Official tongue of {s}: {o}"],
    'developer':        ["{o} built {s}",
                         "The maker of {s} is {o}",
                         "Development of {s} was done by {o}"],
    'field':            ["The profession of {s} is {o}",
                         "{s} practises {o}",
                         "Professional field of {s}: {o}"],
    'famous_for':       ["{s} is known for {o}",
                         "The claim to fame of {s} is {o}",
                         "Renown of {s} comes from {o}"],
    'chairperson':      ["{o} chairs {s}",
                         "The chair of {s} is {o}",
                         "Chairmanship of {s}: {o}"],
    'head_of_gov':      ["{o} heads the government of {s}",
                         "Government leader of {s} is {o}"],
    'head_of_state':    ["{o} is head of state for {s}",
                         "State leadership of {s}: {o}"],
    'music_genre':      ["{s} plays {o} music",
                         "The genre of {s} is {o}"],
    'producer':         ["{o} makes {s}",
                         "The producing company of {s} is {o}"],
    'broadcaster':      ["{o} broadcasts {s}",
                         "The broadcasting network of {s} is {o}"],
    'written_in':       ["{s} appeared in the language {o}",
                         "The language of writing for {s} is {o}"],
    'child':            ["The offspring of {s} is {o}",
                         "{o} is a child of {s}"],
    'pm':               ["{o} serves as prime minister of {s}"],
    'mother_tongue':    ["The native language of {s} is {o}"],
    'head_coach':       ["{o} coaches {s}"],
    'owner':            ["{o} owns {s}"],
}


def perturb_context(ctx, ratio, seed=0):
    """Rewrite a `ratio` share of facts into equivalent unknown surface forms.
    Serial numbers are preserved."""
    rng = random.Random(seed)
    out, changed, total = [], 0, 0
    for line in ctx.split('\n'):
        m = re.match(r'^(\d+)\. (.+?)\.?$', line)
        if not m:
            out.append(line)
            continue
        serial, body = m.group(1), m.group(2).rstrip('.')
        rel = subj = obj = None
        for rx, rname in FACT_PATS:
            mm = re.match(rx, body)
            if mm:
                rel, subj, obj = rname, mm.group(1).strip(), mm.group(2).strip()
                break
        total += 1
        if rel and rel in PARAPHRASE and rng.random() < ratio:
            tmpl = rng.choice(PARAPHRASE[rel])
            out.append(f"{serial}. {tmpl.format(s=subj, o=obj)}.")
            changed += 1
        else:
            out.append(line)
    return '\n'.join(out), changed, total


def evaluate(ctx, questions, answers):
    tri, _ = parse_facts(ctx)
    index = build_index(tri)
    subs = sorted({s for (s, _) in index}, key=lambda s: (-len(s), s))
    c = 0
    for q, g in zip(questions, answers):
        if subem(answer(q, index, subs), g):
            c += 1
    return c / len(questions) * 100


if __name__ == '__main__':
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(sys.argv[1] if len(sys.argv) > 1 else 'conflict_resolution.parquet').read().to_pylist()
    targets = [r for r in rows if '_sh_' in r['metadata']['source']]
    ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"{'task':<28}" + ''.join(f"{int(x*100):>7}%" for x in ratios))
    curves = {}
    for r in targets:
        src = r['metadata']['source']
        row = []
        for ratio in ratios:
            pctx, ch, tot = perturb_context(r['context'], ratio, seed=42)
            row.append(evaluate(pctx, r['questions'], r['answers']))
        curves[src] = row
        print(f"{src:<28}" + ''.join(f"{v:>8.1f}" for v in row))
    avg = [sum(curves[s][i] for s in curves) / len(curves) for i in range(len(ratios))]
    print(f"{'AVERAGE':<28}" + ''.join(f"{v:>8.1f}" for v in avg))
    print()
    print(f"Contribution of template knowledge: {avg[0]-avg[-1]:.1f} puan "
          f"({avg[0]:.1f}% -> {avg[-1]:.1f}%)")
