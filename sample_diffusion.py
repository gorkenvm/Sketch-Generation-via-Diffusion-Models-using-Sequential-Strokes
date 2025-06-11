import torch
from example import draw_strokes_gif
from train_diffusion import model, device

@torch.no_grad()
def sample_stroke_sequence(model, seq_len=256, feature_dim=4, steps=50, device='cuda'):
    model.eval()

    x = torch.randn(1, seq_len, feature_dim, device=device)
    for s in range(steps):
        t = torch.tensor(1 - s/steps, device=device).reshape(1,1,1)  
        x_pred = model((1-t)*x + t*torch.randn_like(x))  

        x = x_pred
    return x[0].cpu().numpy()

def seq_to_strokes(seq, threshold=0.5):
    """
    Model sample'ını QuickDraw stroke formatına [ [[x],[y]], ... ] çevirir.
    """
    x = seq[:,0]
    y = seq[:,1]
    starts = seq[:,2] > threshold 
    x = (x * 255).clip(0, 255).astype(int)
    y = (y * 255).clip(0, 255).astype(int)
    strokes = []
    curr_x = []
    curr_y = []
    for xi, yi, si in zip(x, y, starts):
        if si and len(curr_x)>0:     
            strokes.append([curr_x, curr_y])
            curr_x = []
            curr_y = []
        curr_x.append(int(xi))
        curr_y.append(int(yi))
    if len(curr_x)>0:
        strokes.append([curr_x, curr_y])
    return strokes


sample_seq = sample_stroke_sequence(model, seq_len=256, feature_dim=4, steps=50, device=device)
fake_strokes = seq_to_strokes(sample_seq)

draw_strokes_gif(fake_strokes, "sample_fake.gif")