### **📌 Graph Neural Networks (GNN)**
GNNs are designed to operate on **graph-structured data**, such as social networks, molecular graphs, and recommendation systems.

#### **📍 Key Concepts**
1. **Graph Representation**  
   - A graph consists of **nodes (vertices) and edges (connections)**.
   - Each **node** can have associated **features** (e.g., properties of a molecule).
  
2. **Message Passing**  
   - Each node updates its representation based on its **neighbors**.
   - This is done through **aggregation functions** (sum, mean, max).

3. **Graph Convolutional Network (GCN)**  
   - Works like a CNN but for graphs.
   - Uses an **adjacency matrix** to propagate information.

---

#### **📍 GNN Example: Node Classification**
We’ll use **PyTorch Geometric** for a basic GCN model.

```python
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print("GNN Training Complete")
```

---
