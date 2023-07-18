import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*1*1, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = EmotionRecognitionModel()

from torchsummary import summary
summary(model, (1, 32, 32), device='cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.ImageFolder(root='dataset/kaggle', transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = datasets.ImageFolder(root='dataset/kaggle', transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cpu")
model = model.to(device)

loss_values = []
val_loss_values = []

patience = 5  # Number of epochs to wait for improvement in validation loss
patience_counter = 0  # Counter to keep track of epochs without improvement
best_val_loss = np.inf  # Keep track of best validation loss

for epoch in range(50):  # loop over the dataset multiple times

    # Training phase
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_values.append(running_loss / len(trainloader))
    print('Epoch %d training loss: %.3f' % (epoch + 1, loss_values[-1]))

    # Validation phase
    val_running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
    
    val_loss_values.append(val_running_loss / len(testloader))
    print('Epoch %d validation loss: %.3f' % (epoch + 1, val_loss_values[-1]))

    # Check for improvement in validation loss
    if val_loss_values[-1] < best_val_loss:  # Loss improved
        best_val_loss = val_loss_values[-1]
        patience_counter = 0  # Reset the counter
    else:  # Loss did not improve
        patience_counter += 1  # Increment the counter

    if patience_counter >= patience:
        print(f"Stopping training because validation loss has not improved in {patience} epochs.")
        break

print('Finished Training')