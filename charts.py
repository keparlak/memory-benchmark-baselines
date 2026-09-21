"""Charts for the post - generated from the numbers, never drawn by an image model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8984'

# validated categorical palette
C_NOMEM = '#2a78d6'   # slot 1  mavi   — hafiza katmani yok
C_RETR = '#1baf7a'    # slot 3  aqua   — basit erisim
C_MEM = '#eb6834'     # slot 2  turuncu— hafiza urunleri
C_OURS = '#4a3aa7'    # slot 7  violet — bu yazinin taban cizgisi

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})

# --------------------------------------------------------------------- FIGURE 1
DATA = [
    ('Claude-3.7-Sonnet', 89.4, C_NOMEM),
    ('GPT-4o',            87.6, C_NOMEM),
    ('Gemini-2.0-Flash',  84.0, C_NOMEM),
    ('GPT-4o-mini',       82.4, C_NOMEM),
    ('Qwen3-Embedding-4B', 78.0, C_RETR),
    ('MemoRAG',           77.0, C_RETR),
    ('GPT-4.1-mini',      75.6, C_NOMEM),
    ('BM25',              75.4, C_RETR),
    ('Text-Embed-3-Large', 72.4, C_RETR),
    ('Contriever',        70.6, C_RETR),
    ('Text-Embed-3-Small', 70.0, C_RETR),
    ('TF-IDF + kNN  (no LLM, $0)', 68.0, C_OURS),
    ('MemGPT',            67.6, C_MEM),
    ('MIRIX (4.1-mini)',  63.0, C_MEM),
    ('Zep',               62.8, C_MEM),
    ('HippoRAG-v2',       61.4, C_MEM),
    ('RAPTOR',            59.4, C_MEM),
    ('GraphRAG',          39.4, C_MEM),
    ('MIRIX',             38.4, C_MEM),
    ('Cognee',            35.4, C_MEM),
    ('Mem0',              32.4, C_MEM),
    ('Self-RAG',          11.6, C_MEM),
]


def rounded_bar(ax, y, w, color, h=0.62, r=0.9):
    """Thin mark, rounded data-end, anchored to the baseline."""
    ax.add_patch(FancyBboxPatch(
        (0, y - h / 2), max(w - r, 0.1), h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        mutation_aspect=0.06,
        facecolor=color, edgecolor=SURFACE, linewidth=1.4, zorder=3))


def chart1(path='fig1_benchmark.png'):
    fig, ax = plt.subplots(figsize=(11.2, 8.6), dpi=200)
    ys = list(range(len(DATA)))[::-1]
    for y, (name, val, col) in zip(ys, DATA):
        rounded_bar(ax, y, val, col)
        bold = col == C_OURS
        ax.text(val + 1.2, y, f'{val:.1f}', va='center', ha='left',
                fontsize=10.5, color=INK if bold else INK2,
                fontweight='bold' if bold else 'normal', zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DATA], fontsize=10.5, color=INK2)
    for lbl, (name, val, col) in zip(ax.get_yticklabels(), DATA):
        if col == C_OURS:
            lbl.set_color(INK); lbl.set_fontweight('bold')

    # 82.4 reference line - the backbone with no memory layer
    ax.axvline(82.4, color=C_NOMEM, linestyle=(0, (4, 3)), linewidth=1.4,
               alpha=0.5, zorder=1)
    ax.annotate('GPT-4o-mini\nno memory layer\n82.4',
                xy=(82.4, 3.2), xytext=(87.5, 3.2),
                fontsize=9.5, color=C_NOMEM, va='center', ha='left',
                linespacing=1.45,
                arrowprops=dict(arrowstyle='-', color=C_NOMEM, alpha=0.55,
                                linewidth=1.1, shrinkA=0, shrinkB=2))

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.9, len(DATA) - 0.3)
    ax.set_xlabel('Accuracy  (MemoryAgentBench · multi-class classification, avg of 5 tasks)',
                  fontsize=10.5, color=INK2, labelpad=10)
    ax.xaxis.grid(True, color='#e6e5e1', linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#dddcd8')
    ax.tick_params(axis='x', colors=INK3, length=0, labelsize=10)
    ax.tick_params(axis='y', length=0)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_NOMEM, label='No memory layer (long-context)'),
        Patch(facecolor=C_RETR, label='Simple retrieval'),
        Patch(facecolor=C_MEM, label='Memory products'),
        Patch(facecolor=C_OURS, label='This post: TF-IDF + kNN, zero LLM'),
    ], loc='lower left', bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
        fontsize=10, labelcolor=INK2, handlelength=1.1, handleheight=1.1,
        columnspacing=2.2, handletextpad=0.7)

    fig.suptitle('Memory products score below using no memory at all',
                 x=0.005, y=1.10, ha='left', fontsize=17.5, color=INK,
                 fontweight='bold')
    fig.text(0.005, 1.035,
             'A TF-IDF + k-NN baseline with no LLM beats eight published memory systems.\n'
             'Source: MemoryAgentBench (ICLR 2026), Table 7 — baseline added.',
             ha='left', va='top', fontsize=11, color=INK2, linespacing=1.5)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.38)
    print('yazildi:', path)


# --------------------------------------------------------------------- FIGURE 2
def chart2(path='fig2_robustness.png'):
    x = [0, 10, 25, 50, 75, 100]
    y = [89.0, 80.8, 68.2, 47.0, 23.8, 0.0]
    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=200)

    ax.axhline(54, color=INK3, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
    ax.text(103, 56.5, 'best published system on this task  54',
            fontsize=10, color=INK2, va='bottom', ha='right')

    ax.plot(x, y, color=C_OURS, linewidth=2.0, zorder=4,
            solid_capstyle='round')
    ax.scatter(x, y, s=64, color=C_OURS, zorder=5,
               edgecolor=SURFACE, linewidth=2)
    # 47.0 sits on the dashed line - move it below
    offsets = {0: (0, 14), 10: (0, 14), 25: (0, 14),
               50: (0, -24), 75: (0, 14), 100: (0, 14)}
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                    xytext=offsets[xi], ha='center', fontsize=10.5, color=INK2)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-8, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}%' for v in x])
    ax.set_xlabel('Share of facts rewritten into surface forms the parser does not know',
                  fontsize=10.5, color=INK2, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=10.5, color=INK2, labelpad=10)
    ax.yaxis.grid(True, color='#e6e5e1', linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#dddcd8')
    ax.tick_params(colors=INK3, length=0, labelsize=10)

    fig.suptitle('100% of the score is template knowledge',
                 x=0.005, y=1.14, ha='left', fontsize=17.5, color=INK,
                 fontweight='bold')
    fig.text(0.005, 1.055,
             'Paraphrase the facts and the deterministic parser collapses to zero.\n'
             'This is a benchmark critique, not a method.',
             ha='left', va='top', fontsize=11, color=INK2, linespacing=1.5)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.38)
    print('yazildi:', path)


if __name__ == '__main__':
    chart1(); chart2()
