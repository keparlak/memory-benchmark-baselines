"""Fetch the public MemoryAgentBench parquet splits. No auth needed."""
import urllib.request, sys

BASE = 'https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/parquet/default'
SPLITS = {
    'Conflict_Resolution': 'conflict_resolution.parquet',
    'Test_Time_Learning': 'Test_Time_Learning.parquet',
}


def fetch(split, out):
    url = f'{BASE}/{split}/0.parquet'
    print(f'{split} ... ', end='', flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'memory-benchmark-baselines'})
    data = urllib.request.urlopen(req, timeout=600).read()
    with open(out, 'wb') as f:
        f.write(data)
    print(f'{len(data)/1e6:.1f} MB -> {out}')


if __name__ == '__main__':
    want = sys.argv[1:] or list(SPLITS)
    for s in want:
        if s not in SPLITS:
            print(f'unknown split: {s}  (options: {", ".join(SPLITS)})')
            continue
        fetch(s, SPLITS[s])
    print('done')
