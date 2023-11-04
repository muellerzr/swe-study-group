# Code was borrowed from the official PyTorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets as dset
from torchvision import transforms as tfms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.d1 = nn.Dropout(0.25)
        self.d2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.max = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        x = self.fc2(self.d2(self.act(self.fc1(self.d1(self.flatten(x))))))
        return self.max(x)

def train(m,d,tl,opt):
    m.train()
    for x,y in enumerate(tl):
        x,y = x.to(d), y.to(d)
        opt.zero_grad()
        out = m(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        opt.step()

def evaluate(m,d,el):
    m.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in el:
            x,y = x.to(d), y.to(d)
            out = m(x)
            loss += F.nll_loss(out, y, reduction='sum').item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(el.dataset)
    acc = correct / len(el.dataset)
    print(f'Average loss: {loss:.4f}, Accuracy: {correct}/{len(el.dataset)} ({acc:.2f}%)')

def main():
    tfm = tfms.Compose([tfms.ToTensor(), tfms.Normalize((0.1307,), (0.3081,))])
    dset1 = dset.MNIST('../data', train=True, download=True, transform=tfm)
    dset2 = dset.MNIST('../data', train=False, transform=tfm)
    tl = torch.utils.data.DataLoader(dset1, batch_size=64, shuffle=True)
    el = torch.utils.data.DataLoader(dset2, batch_size=128)
    m = Net()
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m.to(d)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    sched = StepLR(opt, step_size=1, gamma=0.7)
    for epoch in range(0, 10):
        train(m,d,tl,opt)
        evaluate(m,d,el)
        sched.step()

if __name__ == "__main__":
    main()