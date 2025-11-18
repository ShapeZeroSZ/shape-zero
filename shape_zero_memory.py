# Shape Zero Master-Fractal Indexing – November 2025
# Author: Lucas Shouse
# GitHub: https://github.com/ShapeZeroSZ/shape-zero
# License: Free for research/education/personal · Commercial requires permission

import torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt
from torchvision import datasets, transforms

class ShapeZeroMemory(nn.Module):
    def __init__(self, alpha=0.8, levels=6):
        super().__init__()
        self,alpha = alpha; self.levels = levels
        self.projectors = nn.ModuleList([nn.Linear(784, 128//(2**i), bias=False) for i in range(levels)])
        
    def fractional_fft(self, x):
        b = x.shape[0]
        x = x.view(b,1,28,28)
        fft = torch.fft.fft2(x)
        f = torch.fft.fftfreq(28, device=x.device)
        gy,gx = torch.meshgrid(f,f,indexing='ij')
        r = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return torch.fft.ifft2(fft * r**(self.alpha-1)).real.view(b,-1)
    
    def forward(self, x):
        x = self.fractional_fft(x)
        residuals, recon, cur = [], torch.zeros_like(x), x
        for p in self.projectors:
            code = p(cur)
            recon_sub = F.linear(code, p.weight.t())
            residuals.append(code)
            recon += recon_sub
            cur -= recon_sub
        return recon, residuals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShapeZeroMemory().to(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=8, shuffle=True)

# ←←← Flip to True for the 30-second trained version (recommended!)
TRAIN = True

if TRAIN:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    model.train()
    for i,(img,_) in enumerate(train_loader):
        if i > 800: break
        img = img.to(device)
        recon,_ = model(img)
        loss = loss_fn(recon, img)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Training complete – final loss {loss.item():.6f}")

model.eval()
img,_ = next(iter(loader)); img = img.to(device)
recon, codes = model(img)

plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,8,i+1); plt.imshow(img[i].cpu().view(28,28), cmap='gray'); plt.axis('off')
    plt.subplot(2,8,i+9); plt.imshow(recon[i].detach().cpu().view(28,28), cmap='gray'); plt.axis('off')
plt.suptitle("Shape Zero – Top: Original | Bottom: Reconstruction")
plt.tight_layout(); plt.savefig("shape_zero_demo.png"); plt.show()

print(f"Compression: {784*8 / sum(c.numel() for c in codes):.1f}×")
