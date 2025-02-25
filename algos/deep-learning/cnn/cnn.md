### **📌 Convolutional Neural Networks (CNN) Refresher**
CNNs are specialized deep learning architectures designed to **process structured grid-like data**, particularly images. They are widely used in **image classification, object detection, and segmentation**.

#### **📍 Key Components of CNNs**
1. **Convolutional Layers**  
   - Perform **feature extraction** using convolutional filters (kernels).
   - Helps detect **edges, textures, shapes**, and **high-level patterns**.
  
2. **Pooling Layers**  
   - Downsample the feature maps to **reduce dimensionality**.
   - Common types: **MaxPooling, AveragePooling**.
  
3. **Fully Connected (FC) Layers**  
   - Used for **final classification**.
   - Converts **feature maps** into **flattened vector**.

4. **Activation Functions**  
   - Typically **ReLU** (Rectified Linear Unit) is used to introduce **non-linearity**.

---

#### **📍 Simple CNN in PyTorch**
We’ll create a CNN for classifying images from the **MNIST dataset**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training Complete")
```

---
