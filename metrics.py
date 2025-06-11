import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import numpy as np
from metrics import compute_fid, compute_kid
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_fid(real_imgs, fake_imgs):
    """
    real_imgs, fake_imgs: [N, 3, H, W] torch tensor ya da numpy (0-1 arası veya 0-255)
    """
    if isinstance(real_imgs, np.ndarray):
        real_imgs = torch.from_numpy(real_imgs)
    if isinstance(fake_imgs, np.ndarray):
        fake_imgs = torch.from_numpy(fake_imgs)
    # Images should be [N, 3, H, W], dtype=float, 0-255 or 0-1
    metric = FrechetInceptionDistance(normalize=True, feature=2048)
    metric.update(real_imgs, real=True)
    metric.update(fake_imgs, real=False)
    fid = metric.compute().item()
    return fid

def compute_kid(real_imgs, fake_imgs):
    if isinstance(real_imgs, np.ndarray):
        real_imgs = torch.from_numpy(real_imgs)
    if isinstance(fake_imgs, np.ndarray):
        fake_imgs = torch.from_numpy(fake_imgs)
    metric = KernelInceptionDistance(subset_size=50)
    metric.update(real_imgs, real=True)
    metric.update(fake_imgs, real=False)
    kid_mean, kid_std = metric.compute()
    return kid_mean.item(), kid_std.item()






def stroke_to_img_array(strokes, img_size=64):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(img_size/32, img_size/32), dpi=32)
    plt.close(fig)
    for stroke in strokes:
        x, y = stroke
        ax.plot(x, y, linewidth=2, color='black')
    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGB al
    img = img[..., 0] * 0.299 + img[...,1]*0.587 + img[...,2]*0.114   # gri yap
    img = 255 - img  # Kağıt beyaz, kalem siyah
    img = np.clip(img, 0, 255)
    img = np.expand_dims(img, 0)      # [1, H, W], gri
    img = np.repeat(img, 3, axis=0)   # [3, H, W], RGB
    img = img.astype(np.uint8)        # ----- EN ÖNEMLİSİ -----
    return img





def evaluate_fid_kid(real_strokes_list, fake_strokes_list, img_size=64):
    real_imgs = np.stack([stroke_to_img_array(s, img_size=img_size) for s in real_strokes_list])
    fake_imgs = np.stack([stroke_to_img_array(s, img_size=img_size) for s in fake_strokes_list])
    real_imgs = torch.from_numpy(real_imgs).to(torch.uint8)
    fake_imgs = torch.from_numpy(fake_imgs).to(torch.uint8)
    print(f"Gerçek images shape: {real_imgs.shape}, dtype: {real_imgs.dtype}")
    print(f"Fake images shape: {fake_imgs.shape}, dtype: {fake_imgs.dtype}")
    fid = compute_fid(real_imgs, fake_imgs)
    kid_mean, kid_std = compute_kid(real_imgs, fake_imgs)
    print(f"FID: {fid:.2f} | KID: {kid_mean:.4f} ± {kid_std:.4f}")
    return fid, kid_mean, kid_std