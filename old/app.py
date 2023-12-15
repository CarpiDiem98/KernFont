from ..library.resize import resize_cv
from ..library.font_to_matrix import takeKernDict, matrix_kern

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

source_OTF = '../Dataset/OTF'
source_UFO = '../Dataset/UFO'

scale_value = 32

print('Creazione dataframe')


class Font(Dataset):
    def __init__(self, source_OTF, source_UFO, scale_value):
        self.source_otf = source_OTF
        self.source_ufo = source_UFO
        self.scale_value = scale_value

    def __getitem__(self, index):
        let1, let2, value = matrix_kern(
            self.source_otf, takeKernDict(self.source_ufo), self.scale_value)
        let1 = resize_cv((let1.values).tolist(), scale_value)
        let2 = resize_cv((let2.values).tolist(), scale_value)
        value = value.to_numpy(dtype=float)
        return let1[index], let2[index], value[index]

    def __len__(self):
        return len(takeKernDict(self.source_ufo))


class MultiInputRegressionNet(nn.Module):
    def __init__(self):
        super(MultiInputRegressionNet, self).__init__()
        # branch1
        self.conv1_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # branch 2
        self.conv1_2 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x1, x2):
        x1 = self.conv1_1(x1)
        x1 = self.relu1_1(x1)
        x1 = self.pool1_1(x1)

        x2 = self.conv1_2(x2)
        x2 = self.relu1_2(x2)
        x2 = self.pool1_2(x2)

        x = torch.cat((x1, x2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


print('Passaggio dataset to Font Class')
# definizione del dataset
dataset = Font(source_OTF, source_UFO)

# Dividi il dataset in training e test utilizzando random_split
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(
    val_dataset, batch_size=64, shuffle=True, num_workers=2)

model = MultiInputRegressionNet()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define the number of epochs
num_epochs = 30
best_val_loss = float("inf")  # initialize with a large value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print('Inizio fase di addestramento')
# Training loop
for epoch in range(num_epochs):
    # Initialize the loss for this epoch
    train_loss = 0.0
    val_loss = 0.0

    # Training
    for idx, (x1, x2, y) in enumerate(train_dataloader):
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x1[idx], x2[idx])

        # Compute the loss
        loss = loss_fn(outputs, y.unsqueeze(1)[idx].float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the training loss
        train_loss += loss.item()

    # Validation
    with torch.no_grad():
        for idx, (x1, x2, y) in enumerate(val_dataloader):
            # Forward pass
            outputs = model(x1[idx], x2[idx])

            # Compute the loss
            loss = loss_fn(outputs.float(), y.unsqueeze(1)[idx].float())

            # Accumulate the validation loss
            val_loss += loss.item()

    # Print the epoch loss
    train_loss = train_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)
    print('Epoch {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(
        epoch+1, train_loss, val_loss))

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print("Model saved with validation loss:{:.4f}".format(best_val_loss))
