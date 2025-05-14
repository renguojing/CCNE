import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import numpy as np

class GCNEncoder(torch.nn.Module):
    """GCN组成的编码器"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    """解码器，用向量内积表示重建的图结构"""

    def forward(self, z, edge_index, sigmoid=True):
        """
        参数说明：
        z: 节点表示
        edge_index: 边索引，也就是节点对
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class GAE(torch.nn.Module):
    """图自编码器。
    """

    def __init__(self, encoder=GCNEncoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        """编码功能"""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """解码功能"""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """计算正边和负边的二值交叉熵

        参数说明
        ----
        z: 编码器的输出
        pos_edge_index: 正边的边索引
        neg_edge_index: 负边的边索引
        """
        EPS = 1e-15  # EPS是一个很小的值，防止取对数的时候出现0值

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()  # 正样本的损失函数

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0))  # 负采样
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()  # 负样本的损失函数

        return pos_loss + neg_loss

def get_embedding(x, edge_index, dim, lr=0.001, epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    input = x.shape[1]
    model = GAE(GCNEncoder(input, dim))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch: {:03d}, Loss_train: {:.8f}'.format(epoch, loss))
    model.eval()
    embedding=model.encode(x, edge_index)
    embedding=embedding.detach().cpu()
    return embedding
