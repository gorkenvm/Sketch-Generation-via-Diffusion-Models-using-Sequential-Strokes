import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio

def draw_strokes_gif_with_stroke_counter(strokes, gif_path, img_size=256, duration=0.35):
    """
    - strokes: QuickDraw format [[x],[y]], ... listesi
    - gif_path: Kaydedilecek dosya
    """
    images = []
    total_strokes = len(strokes)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=img_size//3)
    plt.close(fig)

    curr_strokes = []
    for s_idx, stroke in enumerate(strokes):
        curr_strokes.append(stroke)
        ax.clear()
        for idx, s in enumerate(curr_strokes):
            x, y = s
            ax.plot(x, y, linewidth=2, color='black')
        ax.axis('off')
        ax.set_aspect('equal', adjustable='datalim')
        ax.text(0, 20, f"Stroke: {s_idx+1}/{total_strokes}", fontsize=12, ha='left', va='top')
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(img.copy())
    imageio.mimsave(gif_path, images, duration=duration)