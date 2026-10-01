# FakE ?!

> FakeE is a DDPM with a U-Net backbone, trained on MNIST.

A simple diffusion model implemented in PyTorch.

<img width="1500" height="300" alt="unets" src="https://github.com/user-attachments/assets/e6e388fc-a185-4e41-a138-b04b8108a6f1" />

## Requirements

```python
pip install -r requirements.txt
```
## Train
```python
python train.py
```

## Generate

```python
python test.py
```

The model starts from random noise and progressively denoises it to generate a new MNIST digit.

## Project Structure

```text
FakeE/
├── model/
│   └── FakeE.pth
├── model.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
