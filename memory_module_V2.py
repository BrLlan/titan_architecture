import torch
import torch.nn as nn


class TitanMLPMemory(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()

        # Layers of the memory MLP (updated every inference step)
        self.W1 = nn.Linear(d_in, d_hidden, bias=False)
        self.W2 = nn.Linear(d_hidden, d_out, bias=False)

        # Momentum buffers for S_t ("update buffer" or "momentum-like hidden state")
        self.register_buffer("S1", torch.zeros_like(self.W1.weight))
        self.register_buffer("S2", torch.zeros_like(self.W2.weight))

        # "learning rate"
        self.theta = nn.Parameter(torch.tensor(0.01))
        # decay coefficient for momentum (by how much does momentumd ecrease) 
        self.eta   = nn.Parameter(torch.tensor(0.9))

    def forward(self, k):
        # k: [batch, d_in]
        #h1 = torch.sigmoid(self.W1(k))
        h1 = self.W1(k) * torch.sigmoid(self.W1(k))      # SiLU
        y  = self.W2(h1)
        return y, h1

    def update_memory(self, k, v):
        # forward part
        # (always called after forward, so no extra forward call needed?
        # pr combine the two?)
        y, h1 = self.forward(k)

        # error calcualtion
        e = y - v                          # [batch, d_out]

        # Update layers and biases
        # gradient for layer 2
        grad2 = e.T @ h1                   # [d_out, d_hidden]

        # Momentum update
        self.S2 = self.eta * self.S2 - self.theta * grad2

        # Apply update
        with torch.no_grad():
            self.W2.weight += self.S2

        # layer 1 update
        # delta1 = (e @ W2) * f'(z1)

        #SILU derivative. Needs input self.W1(k)
        sig = torch.sigmoid(self.W1(k))
        silu_prime = sig + self.W1(k) * sig * (1 - sig)

        delta1 = (e @ self.W2.weight) * silu_prime
        #delta1 = (e @ self.W2.weight) * (h1 * (1 - h1))

        grad1 = delta1.T @ k               # [d_hidden, d_in]

        self.S1 = self.eta * self.S1 - self.theta * grad1
        
        # Apply update
        with torch.no_grad():
            self.W1.weight += self.S1
