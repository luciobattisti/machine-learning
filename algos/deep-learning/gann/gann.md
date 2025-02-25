### **📌 Generative Adversarial Networks (GANs)**
GANs consist of two networks **competing** with each other:
1. **Generator (G)** - Creates fake images from random noise.
2. **Discriminator (D)** - Distinguishes between real and fake images.

#### **📍 How GANs Work**
1. The **Generator** takes random noise and generates an image.
2. The **Discriminator** checks if the image is real or fake.
3. The Generator tries to **fool** the Discriminator.
4. The process repeats until the Generator produces realistic images.

---

#### **📍 GAN Implementation in PyTorch**
We’ll train a **simple GAN** on the **MNIST dataset**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 784))

# Initialize models
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# Training Loop
for epoch in range(10):
    for real_images, _ in trainloader:
        batch_size = real_images.size(0)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        optimizer_D.zero_grad()
        outputs = D(real_images)
        loss_real = criterion(outputs, real_labels)

        noise = torch.randn(batch_size, 100)
        fake_images = G(noise)
        outputs = D(fake_images.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = D(fake_images)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

print("GAN Training Complete")
```

---
