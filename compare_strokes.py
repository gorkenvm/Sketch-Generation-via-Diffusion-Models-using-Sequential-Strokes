import matplotlib.pyplot as plt

def plot_real_vs_fake(real_strokes, fake_strokes, out_fp="compare.png"):
    fig, ax = plt.subplots(figsize=(4,4))
    for stroke in real_strokes:
        x, y = stroke
        ax.plot(x, y, linewidth=2, color='blue', alpha=0.6, label="Gerçek")
    for stroke in fake_strokes:
        x, y = stroke
        ax.plot(x, y, linewidth=2, color='red', alpha=0.8, label="Fake")
    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    # Sadece ilk iterasyonda label göster
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = {l:h for h,l in zip(handles,labels)}
        ax.legend(unique.values(), unique.keys(), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.show()
