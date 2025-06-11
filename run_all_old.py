import os
import torch
from dataset_utils import QuickDrawStrokeDataset
from torch.utils.data import DataLoader
from diffusion_model import DiffusionMLP
from trainer_utils import diffusion_train_step
from sample_diffusion import sample_stroke_sequence, seq_to_strokes
from stroke_numarali_animasyon_gif import draw_strokes_gif_with_stroke_counter
from metrics import evaluate_fid_kid
import ndjson
import json


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

categories = ['bus', 'cat', 'rabbit']
max_seq_len = 256
n_train_epoch = 40      # Daha fazla epoch ile kalite artar
batch_size = 32
n_samples = 50         # Her kategoriden üreteceğin fake sample sayısı

for cat in categories:
    print(f"\n==== {cat.upper()} ====")
    ds = QuickDrawStrokeDataset(
        f'subset/{cat}/{cat}.ndjson',
        f'subset/{cat}/indices.json',
        split='train',
        max_seq_len=max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DiffusionMLP(seq_len=max_seq_len, features=4, hidden_dim=256, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Eğitim başlıyor...")
    for epoch in range(n_train_epoch):
        for batch in loader:
            loss = diffusion_train_step(model, batch, optimizer, device)
        print(f"Epoch {epoch+1}: Loss: {loss:.4f}")

    print("Modelden örnek çizimler (fake) üretiliyor...")
    fake_strokes = []
    for i in range(n_samples):
        seq = sample_stroke_sequence(model, seq_len=max_seq_len, feature_dim=4, steps=50, device=device)
        fake_strokes.append(seq_to_strokes(seq))
    print("Fake sample sayısı:", len(fake_strokes))


    draw_strokes_gif_with_stroke_counter(fake_strokes[0], f'{cat}_fake_sample_animation.gif')
    print(f"Animasyonlu fake örnek kaydedildi: {cat}_fake_sample_animation.gif")


    print("Gerçek test çizimleri yükleniyor...")
    with open(f'subset/{cat}/{cat}.ndjson', 'r') as f:
        all_drawings = ndjson.load(f)
    with open(f'subset/{cat}/indices.json', 'r') as f:
        indices = json.load(f)
    real_strokes = [all_drawings[i]['drawing'] for i in indices['test'][:n_samples]]

    print("FID/KID skorları hesaplanıyor...")
    fid, kid_mean, kid_std = evaluate_fid_kid(real_strokes, fake_strokes, img_size=64)
    print(f"[{cat.upper()}] FID: {fid:.2f} | KID: {kid_mean:.4f} ± {kid_std:.4f}\n")

print("PIPELINE TAMAMLANDI.")