import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

training_size = int(0.8 * len(dataset))
test_size = len(dataset) - training_size

training_set, test_set = torch.utils.data.random_split(dataset, [training_size, test_size])
train_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)


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

    def forward(self, x, debug=False):
        if debug:
            print("Input shape:", x.shape)

        x = self.relu(self.conv1(x))
        if debug:
            print("After conv1:", x.shape)

        x = self.pool(x)
        if debug:
            print("After pool1:", x.shape)

        x = self.relu(self.conv2(x))
        if debug:
            print("After conv2:", x.shape)

        x = self.pool(x)
        if debug:
            print("After pool2:", x.shape)

        x = x.view(-1, 64 * 7 * 7)  # Flatten
        if debug:
            print("After Flatten:", x.shape)

        x = self.relu(self.fc1(x))
        if debug:
            print("After fc1:", x.shape)

        x = self.fc2(x)
        if debug:
            print("After fc2 (output):", x.shape)

        return x

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 10
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model.forward(images, debug=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        input("Next?")

print("Training Complete")
