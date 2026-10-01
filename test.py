import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import FakeE, TimeEmbedding

T = 1000
IMAGE_SIZE = 64
MODEL_PATH = "./model/FakeE.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


beta = torch.linspace(1e-4, 0.02, T, device=device)

alpha = 1.0 - beta

alpha_bar = torch.cumprod(alpha, dim=0)


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


model = FakeE().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

print("Model loaded!")


x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)


show_steps = [999, 900, 700, 500, 300, 100, 0]

results = []

with torch.no_grad():
    for t in reversed(range(T)):
        timestep = torch.tensor([t], device=device, dtype=torch.long)

        t_emb = TimeEmbedding(timestep).forward(timestep)

        predicted_noise = model(x, t_emb)

        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x)

            x = x + torch.sqrt(beta_t) * noise

        if t in show_steps:
            results.append((t, x.cpu().squeeze().clone()))


fig, axes = plt.subplots(1, len(results), figsize=(15, 3))

for ax, (t, image) in zip(axes, results):
    ax.imshow(image, cmap="gray", vmin=-1, vmax=1)

    ax.set_title(f"t = {t}")
    ax.axis("off")

plt.tight_layout()
plt.show()
