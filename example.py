import os
import ndjson
import random
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import imageio

def load_drawings(category_path):
    cat = os.path.basename(category_path)
    ndjson_file = os.path.join(category_path, f"{cat}.ndjson")
    if not os.path.exists(ndjson_file):
        raise FileNotFoundError(f"{ndjson_file} bulunamadı!")
    with open(ndjson_file, "r") as f:
        data = ndjson.load(f)
    print(f"{cat} kategorisinden {len(data)} çizim yüklendi.")
    return data

def plot_strokes(strokes, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for stroke in strokes:
        x, y = stroke
        ax.plot(x, y, linewidth=2, color='black')
    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    return ax

def draw_strokes_gif(strokes, gif_path, img_size=256, duration=0.35):
    images = []
    fig, ax = plt.subplots(figsize=(3,3), dpi=img_size//3)
    plt.close(fig)
    for i in range(1, len(strokes)+1):
        ax.clear()
        plot_strokes(strokes[:i], ax=ax)
        fig.canvas.draw()

        img = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(img.copy())

    imageio.mimsave(gif_path, images, duration=duration)


categories = ["bus", "cat", "rabbit"]
project_root = os.getcwd()  

for cat in categories:
    cat_path = os.path.join(project_root, "subset", cat)
    data = load_drawings(cat_path)
    N = len(data)
    print(f"{cat} kategorisinden ilk 2 ve rastgele 1 çizimi GIF'e dönüştürüyorum...")

    for i in range(2):
        strokes = data[i]['drawing']
        draw_strokes_gif(strokes, f"{cat}_sample_{i}.gif")

    random_idx = random.randint(0, N-1)
    strokes = data[random_idx]['drawing']
    draw_strokes_gif(strokes, f"{cat}_random.gif")

print("GIF dosyaların başarıyla oluşturuldu.")