"""
FactConsolidation baseline for MemoryAgentBench - zero LLM calls.

Parse the fact list into (serial, subject, relation, object) triples, index by
(subject, relation) keeping the HIGHEST serial, then resolve each question -
including multi-hop "the A of the B of X" chains - against that index.

Finding: within the questions the parser covers, single-hop accuracy is
96.6-100%. The deterministic newest-serial policy is essentially exact; all the
loss sits in semantic matching. See robustness_paraphrase.py before quoting the
headline number.

Data:   https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
Metric: SubEM (substring exact match), as in MAB.
"""
import re, sys
from collections import defaultdict

FACT_PATS = [
    (r'^The chairperson of (.+) is (.+)$', 'chairperson'),
    (r'^The director of (.+) is (.+)$', 'director'),
    (r'^The author of (.+) is (.+)$', 'author'),
    (r'^The capital of (.+) is (.+)$', 'capital'),
    (r'^The headquarters of (.+) is located in the city of (.+)$', 'hq_city'),
    (r'^The chief executive officer of (.+) is (.+)$', 'ceo'),
    (r'^The univeristy where (.+) was educated is (.+)$', 'educated_at'),
    (r'^The name of the current head of the (.+) government is (.+)$', 'head_of_gov'),
    (r'^The name of the current head of state in (.+) is (.+)$', 'head_of_state'),
    (r'^The Prime Minister of (.+) is (.+)$', 'pm'),
    (r'^The type of music that (.+) plays is (.+)$', 'music_genre'),
    (r'^The official language of (.+) is (.+)$', 'official_language'),
    (r'^The company that produced (.+) is (.+)$', 'producer'),
    (r'^The origianl broadcaster of (.+) is (.+)$', 'broadcaster'),
    (r'^The original broadcaster of (.+) is (.+)$', 'broadcaster'),
    (r'^The mother tongue of (.+) is (.+)$', 'mother_tongue'),
    (r'^The head coach of (.+) is (.+)$', 'head_coach'),
    (r'^The occupation of (.+) is (.+)$', 'field'),
    (r"^(.+)'s child is (.+)$", 'child'),
    (r'^(.+) was born in the city of (.+)$', 'born_city'),
    (r'^(.+) died in the city of (.+)$', 'died_city'),
    (r'^(.+) plays the position of (.+)$', 'position'),
    (r'^(.+) is located in the continent of (.+)$', 'continent'),
    (r'^(.+) worked in the city of (.+)$', 'worked_city'),
    (r'^(.+) works in the field of (.+)$', 'field'),
    (r'^(.+) is married to (.+)$', 'spouse'),
    (r'^(.+) is associated with the sport of (.+)$', 'sport'),
    (r'^(.+) was founded by (.+)$', 'founded_by'),
    (r'^(.+) was founded in the city of (.+)$', 'founded_city'),
    (r'^(.+) was created in the country of (.+)$', 'created_country'),
    (r'^(.+) was created by (.+)$', 'created_by'),
    (r'^(.+) was developed by (.+)$', 'developer'),
    (r'^(.+) is a citizen of (.+)$', 'citizen'),
    (r'^(.+) was performed by (.+)$', 'performer'),
    (r'^(.+) speaks the language of (.+)$', 'speaks'),
    (r'^(.+) is employed by (.+)$', 'employer'),
    (r'^(.+) was written in (.+)$', 'written_in'),
    (r'^(.+) is famous for (.+)$', 'famous_for'),
    (r'^(.+) is affiliated with the religion of (.+)$', 'religion'),
    (r'^(.+) is owned by (.+)$', 'owner'),
]

# relation cue lexicon: surface phrase -> relation. Longest match first.
CUES = [
    ('country of citizenship', 'citizen'), ('citizenship of', 'citizen'),
    ('citizen of', 'citizen'), ('nationality of', 'citizen'),
    ('country of origin', 'created_country'), ('birthplace of', 'created_country'),
    ('created in', 'created_country'), ('originally hail', 'created_country'),
    ('come into existence', 'founded_city'), ('founded in', 'founded_city'),
    ('official language', 'official_language'), ('official documents written', 'official_language'),
    ('original language', 'written_in'), ('written in', 'written_in'),
    ('language of', 'speaks'), ('language does', 'speaks'), ('speak', 'speaks'),
    ('mother tongue', 'mother_tongue'),
    ('spouse of', 'spouse'), ('married to', 'spouse'), ('wife of', 'spouse'), ('husband of', 'spouse'),
    ('author of', 'author'), ('wrote', 'author'),
    ('director of', 'director'), ('directed', 'director'),
    ('chairperson of', 'chairperson'),
    ('chief executive officer', 'ceo'), ('ceo of', 'ceo'),
    ('head of state', 'head_of_state'), ('head of the', 'head_of_gov'),
    ('prime minister', 'pm'),
    ('capital of', 'capital'),
    ('headquarters of', 'hq_city'),
    ('educated at', 'educated_at'), ('educated', 'educated_at'), ('university where', 'educated_at'),
    ('type of music', 'music_genre'), ('music that', 'music_genre'),
    ('produced', 'producer'), ('broadcaster of', 'broadcaster'),
    ('pass away', 'died_city'), ('passed away', 'died_city'), ('die', 'died_city'), ('died', 'died_city'),
    ('born', 'born_city'),
    ('position of', 'position'), ('position on a team', 'position'), ('position does', 'position'),
    ('continent', 'continent'),
    ('work in', 'worked_city'), ('worked in', 'worked_city'),
    ('field does', 'field'), ('field of', 'field'), ('occupation of', 'field'),
    ('sport played by', 'sport'), ('sport associated with', 'sport'), ('sport of', 'sport'),
    ('sport is', 'sport'), ('sport was', 'sport'),
    ('founded by', 'founded_by'), ('founder of', 'founded_by'), ('founded', 'founded_by'),
    ('created by', 'created_by'), ('creator of', 'created_by'),
    ('developed by', 'developer'), ('developer of', 'developer'),
    ('performed by', 'performer'), ('performer of', 'performer'),
    ('employed by', 'employer'), ('employer of', 'employer'), ('employs', 'employer'),
    ('famous for', 'famous_for'),
    ('religion associated with', 'religion'), ('religion', 'religion'),
    ('child of', 'child'), ('head coach', 'head_coach'),
    ('owned by', 'owner'), ('owner of', 'owner'),
]
CUES.sort(key=lambda x: -len(x[0]))


def parse_facts(ctx):
    triples, un = [], []
    for line in ctx.split('\n'):
        m = re.match(r'^(\d+)\. (.+?)\.?$', line)
        if not m:
            continue
        serial, body = int(m.group(1)), m.group(2).rstrip('.')
        for rx, rel in FACT_PATS:
            mm = re.match(rx, body)
            if mm:
                triples.append((serial, mm.group(1).strip(), rel, mm.group(2).strip()))
                break
        else:
            un.append(body)
    return triples, un


def build_index(triples):
    """POLICY: (subject, relation) -> highest-serial object. Deterministic."""
    best = {}
    for serial, subj, rel, obj in triples:
        k = (subj.lower(), rel)
        if k not in best or serial > best[k][0]:
            best[k] = (serial, obj)
    return {k: v[1] for k, v in best.items()}


def find_entity(q, subjects_by_len):
    """Find the longest known subject mentioned in the question."""
    ql = q.lower()
    for s in subjects_by_len:
        if s in ql:
            return s
    return None


def cues_in(text):
    """Return relation cues found in text, in position order, non-overlapping."""
    tl = text.lower()
    found, taken = [], []
    for phrase, rel in CUES:
        start = 0
        while True:
            i = tl.find(phrase, start)
            if i < 0:
                break
            if not any(a <= i < b or a < i + len(phrase) <= b for a, b in taken):
                found.append((i, rel))
                taken.append((i, i + len(phrase)))
            start = i + 1
    found.sort()
    return [r for _, r in found]


def answer(q, index, subjects_by_len):
    ent = find_entity(q, subjects_by_len)
    if not ent:
        return None
    ql = q.lower()
    i = ql.find(ent)
    prefix, suffix = q[:i], q[i + len(ent):]
    # innermost cue first: prefix reversed, then suffix in order
    chain = list(reversed(cues_in(prefix))) + cues_in(suffix)
    # drop consecutive duplicates
    dedup = [r for j, r in enumerate(chain) if j == 0 or r != chain[j - 1]]
    cur = ent
    for rel in dedup[:6]:
        nxt = index.get((cur.lower(), rel))
        if nxt is None:
            return None if cur == ent else cur
        cur = nxt
    return None if cur == ent else cur


def subem(pred, golds):
    if pred is None:
        return False
    p = pred.lower().strip()
    return any(g.lower().strip() in p or p in g.lower().strip() for g in golds if g)


def run(parquet_path='conflict_resolution.parquet'):
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(parquet_path).read().to_pylist()
    out = []
    for r in rows:
        triples, un = parse_facts(r['context'])
        index = build_index(triples)
        subjects_by_len = sorted({s for (s, _) in index}, key=lambda s: (-len(s), s))
        correct = nores = 0
        misses = []
        for q, gold in zip(r['questions'], r['answers']):
            p = answer(q, index, subjects_by_len)
            if p is None:
                nores += 1
            if subem(p, gold):
                correct += 1
            elif len(misses) < 3:
                misses.append((q, gold, p))
        n = len(r['questions'])
        cov = n - nores
        out.append(dict(source=r['metadata']['source'], n=n, correct=correct,
                        acc=correct / n * 100, cov=cov,
                        acc_on_cov=(correct / cov * 100) if cov else 0.0,
                        misses=misses))
    return out


if __name__ == '__main__':
    # Gold answers contain non-ASCII names (e.g. 'Kūkai'); Windows consoles
    # default to a legacy code page and would crash on print.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    res = run(sys.argv[1] if len(sys.argv) > 1 else 'conflict_resolution.parquet')
    print(f"{'task':<32}{'n':>4}{'correct':>9}{'acc%':>7}{'covered':>9}{'acc_on_covered%':>17}")
    for r in res:
        print(f"{r['source']:<32}{r['n']:>4}{r['correct']:>7}{r['acc']:>7.1f}{r['cov']:>8}{r['acc_on_cov']:>15.1f}")
    sh = [r for r in res if '_sh_' in r['source']]
    mh = [r for r in res if '_mh_' in r['source']]
    print(f"\nsingle-hop avg: {sum(r['acc'] for r in sh)/len(sh):.1f}%   (MAB best reported: 54%)")
    print(f"multi-hop  avg: {sum(r['acc'] for r in mh)/len(mh):.1f}%   (MAB all 22 systems: <=7%)")
    print("\n--- sample misses ---")
    for r in res:
        for q, g, p in r['misses'][:2]:
            print(f"[{r['source']}] {q[:100]}\n    gold={g} pred={p}")
