import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges

# Load the Cora dataset (no labels needed)
dataset = Planetoid(root="/tmp/Cora", name="Cora")

# Get the graph data object
data = dataset[0]

# Split edges into train/val/test for reconstruction
data = train_test_split_edges(data)


# Define Graph Autoencoder (GAE)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# Initialize GAE Model
out_channels = 16  # Dimension of learned embeddings
model = GAE(GCNEncoder(dataset.num_features, out_channels))

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
def train():
    model.train()
    optimizer.zero_grad()

    # Encode node embeddings
    z = model.encode(data.x, data.train_pos_edge_index)

    # Decode and compute loss
    loss = model.recon_loss(z, data.train_pos_edge_index)

    # Backpropagation
    loss.backward()
    optimizer.step()
    return loss.item()


# Train for 200 epochs
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Get the learned embeddings
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    print("Learned Node Embeddings Shape:", z.shape)
