"""
Multi-class classification baseline for MemoryAgentBench Test-Time Learning.

TF-IDF over unigrams + bigrams, cosine k-NN (k=15) over the 5,900-8,296 labelled
examples already present in the context. No LLM, no embeddings, no training.

Scores 68.0% average, which places it above eight published memory systems and
below every long-context model and simple retrieval baseline.
"""
import re, sys, math
from collections import Counter, defaultdict


def parse_shots(ctx):
    """'<metin>\\nlabel: <N>' bloklarini (metin, etiket) ciftlerine cevirir."""
    shots = []
    for blk in re.split(r'\n\s*\n', ctx):
        m = re.search(r'^(.*?)\s*\nlabel:\s*(\S+)\s*$', blk.strip(), flags=re.S)
        if m:
            txt = m.group(1).strip()
            lab = m.group(2).strip()
            if txt:
                shots.append((txt, lab))
    return shots


TOKEN = re.compile(r'[a-z0-9]+')


def toks(s, n_char=0):
    t = TOKEN.findall(s.lower())
    out = list(t)
    out += [f'{a}_{b}' for a, b in zip(t, t[1:])]          # bigram
    return out


class TfidfKNN:
    def __init__(self, shots):
        self.docs = []
        self.labels = []
        df = Counter()
        for txt, lab in shots:
            tf = Counter(toks(txt))
            self.docs.append(tf)
            self.labels.append(lab)
            for w in tf:
                df[w] += 1
        N = len(self.docs)
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        # inverted index + norms
        self.inv = defaultdict(list)
        self.norm = []
        for i, tf in enumerate(self.docs):
            s = 0.0
            for w, c in tf.items():
                v = (1 + math.log(c)) * self.idf.get(w, 0.0)
                if v:
                    self.inv[w].append((i, v))
                    s += v * v
            self.norm.append(math.sqrt(s) or 1.0)

    def predict(self, q, k=15):
        tf = Counter(toks(q))
        qs = 0.0
        scores = defaultdict(float)
        for w, c in tf.items():
            if w not in self.idf:
                continue
            qv = (1 + math.log(c)) * self.idf[w]
            qs += qv * qv
            for i, dv in self.inv[w]:
                scores[i] += qv * dv
        if not scores:
            return None
        qn = math.sqrt(qs) or 1.0
        top = sorted(((s / (qn * self.norm[i]), i) for i, s in scores.items()),
                     reverse=True)[:k]
        vote = defaultdict(float)
        for sim, i in top:
            vote[self.labels[i]] += sim
        return max(vote.items(), key=lambda x: x[1])[0]


def subem(pred, golds):
    if pred is None:
        return False
    p = str(pred).strip().lower()
    return any(str(g).strip().lower() == p for g in golds if g is not None)


if __name__ == '__main__':
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(sys.argv[1] if len(sys.argv) > 1
                          else 'Test_Time_Learning.parquet').read().to_pylist()
    print(f"{'task':<42}{'shots':>7}{'classes':>9}{'n':>5}{'correct':>9}{'acc%':>8}")
    accs = []
    for r in rows:
        src = r['metadata']['source']
        if not src.startswith('icl_'):
            continue
        shots = parse_shots(r['context'])
        if not shots:
            print(f'{src:<42} COULD NOT PARSE SHOTS')
            continue
        clf = TfidfKNN(shots)
        c = 0
        for q, g in zip(r['questions'], r['answers']):
            if subem(clf.predict(q), g):
                c += 1
        n = len(r['questions'])
        acc = c / n * 100
        accs.append((src, acc))
        print(f'{src:<42}{len(shots):>7}{len(set(l for _, l in shots)):>7}{n:>5}{c:>7}{acc:>8.1f}')
    if accs:
        print(f'\nICL average: {sum(a for _, a in accs)/len(accs):.1f}%')
