import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load dataset (Cora - Citation Network)
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data, debug=False):
        x, edge_index = data.x, data.edge_index
        if debug:
            print(f"Shape: {x.shape}")
        x = F.relu(self.conv1(x, edge_index))
        if debug:
            print(f"Shape after conv1: {x.shape}")

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if debug:
            print(f"Shape after conv2: {x.shape}")

        return F.log_softmax(x, dim=1)


# Train GCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting Training:")
for epoch in range(200):
    print(f"Epoch: {epoch}")
    model.train()
    optimizer.zero_grad()
    out = model.forward(data, debug=True)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    input("Next?")

print("GNN Training Complete")
