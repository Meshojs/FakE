import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from model import FakeE, SpaceTimeSteps, TimeEmbedding

T = 1000
BATCH_SIZE = 64
EPOCHS = 10
IMAGE_SIZE = 64

t = SpaceTimeSteps(T, 50).forward()

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


def add_noise(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t]
    a_bar = a_bar[:, None, None, None]

    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    return xt, noise


model = FakeE()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)
beta = beta.to(device)
alpha_bar = alpha_bar.to(device)


for epoch in tqdm(range(EPOCHS), desc="Training"):
    total_loss = 0

    for x0, _ in tqdm(
        loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", mininterval=1, dynamic_ncols=True
    ):
        x0 = x0.to(device)

        timestep = torch.randint(0, T, (x0.shape[0],), device=device)

        xt, noise = add_noise(x0, timestep)

        t_emb = TimeEmbedding(timestep).forward(timestep)

        pred_noise = model(xt, t_emb).to(device)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "./model/FakeE.pth")
