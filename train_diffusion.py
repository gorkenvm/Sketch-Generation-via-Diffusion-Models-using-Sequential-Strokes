from dataset_utils import QuickDrawStrokeDataset
from torch.utils.data import DataLoader
import torch
# from diffusion_model import DiffusionMLP
import torch.optim as optim


from diffusion_model import StrokeGRUDiffusion

dataset = QuickDrawStrokeDataset(
    'subset/bus/bus.ndjson',
    'subset/bus/indices.json',
    split='train',
    max_seq_len=256
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))
print(batch['seq'].shape)  # torch.Size([32, 256, 4])




device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StrokeGRUDiffusion(input_dim=4, hidden_dim=256, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

from trainer_utils import diffusion_train_step

epochs = 2
for epoch in range(epochs):
    for batch in loader:
        loss = diffusion_train_step(model, batch, optimizer, device)
    print(f"Epoch {epoch}: Loss {loss:.4f}")