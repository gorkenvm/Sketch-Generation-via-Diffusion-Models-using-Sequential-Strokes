import numpy as np
import torch
from torch.utils.data import Dataset
import ndjson
import json

def strokes_to_seq(strokes):
    """
    Her bir çizimi (x, y, stroke_start, time) dizisine çevirir.
    strokes: [[[x1...],[y1...]], ...]
    Çıkış: (N, 4) numpy array: [x, y, stroke_start, time]
    """
    seq = []
    time = 0
    for i, stroke in enumerate(strokes):
        x_seq, y_seq = stroke
        for j, (x, y) in enumerate(zip(x_seq, y_seq)):
            stroke_start = 1 if j == 0 else 0
            seq.append([x, y, stroke_start, time])
            time += 1
    return np.array(seq, dtype=np.float32)
  
class QuickDrawStrokeDataset(Dataset):
    def __init__(self, ndjson_path, indices_path, split='train', max_seq_len=256):
        with open(ndjson_path, "r") as f:
            self.drawings = ndjson.load(f)
        with open(indices_path, "r") as f:
            idx = json.load(f)[split]
        self.indices = idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        entry = self.drawings[self.indices[i]]
        seq = strokes_to_seq(entry['drawing'])
        if len(seq) >= self.max_seq_len:
            seq = seq[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_amt = self.max_seq_len - len(seq)
            seq = np.pad(seq, ((0, pad_amt), (0, 0)), mode='constant', constant_values=0)
            mask = np.concatenate([np.ones(len(seq)-pad_amt), np.zeros(pad_amt)], axis=0)
        seq[:,0:2] = seq[:,0:2]/255.0  # Normalizasyon (isteğe göre ayarla)
        return {
            'seq': torch.from_numpy(seq),       # [max_seq_len, 4]   x, y, stroke_start, time
            'mask': torch.from_numpy(mask),     # [max_seq_len]
        }