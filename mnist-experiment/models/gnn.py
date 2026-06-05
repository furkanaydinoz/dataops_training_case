import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU()
        )
    def forward(self, nodes, edges):
        B, N, C = nodes.shape
        messages = []
        for i in range(N):
            neighbor_idx = (edges[i] == 1).nonzero(as_tuple=True)[1]
            if len(neighbor_idx) == 0:
                messages.append(torch.zeros(B, C, device=nodes.device))
            else:
                neighbors = nodes[:, neighbor_idx]
                msg = self.message_fn(torch.cat([nodes[:, i:i+1].expand(-1, len(neighbor_idx), -1), neighbors], dim=-1)).mean(dim=1)
                messages.append(msg)
        messages = torch.stack(messages, dim=1)
        updated = self.update_fn(torch.cat([nodes, messages], dim=-1))
        return updated + nodes

class TinyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embed = nn.Linear(784, 64)
        self.mp1 = MessagePassingLayer(64)
        self.mp2 = MessagePassingLayer(64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.num_nodes = 9
        self.edges = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(3):
            for j in range(3):
                self.edges[i*3+j, i*3+j] = 1
                if i > 0: self.edges[i*3+j, (i-1)*3+j] = 1
                if i < 2: self.edges[i*3+j, (i+1)*3+j] = 1
                if j > 0: self.edges[i*3+j, i*3+j-1] = 1
                if j < 2: self.edges[i*3+j, i*3+j+1] = 1

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, 28, 28)
        patches = F.unfold(x, kernel_size=9, stride=9).transpose(1, 2).reshape(B, 9, -1)
        nodes = self.node_embed(patches[:, :self.num_nodes])
        edges = self.edges.unsqueeze(0).expand(B, -1, -1)
        nodes = self.mp1(nodes, edges)
        nodes = self.mp2(nodes, edges)
        pooled = nodes.mean(dim=1)
        return self.classifier(pooled)