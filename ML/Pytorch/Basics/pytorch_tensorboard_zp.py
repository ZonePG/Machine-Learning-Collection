# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
num_epoch = 1

# Load data
train_dataset = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)

batch_size = [256]
learning_rates = [0.001]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for batch_size in batch_size:
    for learning_rate in learning_rates:
        losses = []
        accuracies = []

        step = 0
        model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
        model.train()
        writer = SummaryWriter(f'runs/MINIST/MiniBatchSize {batch_size} LR {learning_rate}')
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Train Network
        for epoch in range(num_epoch):
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for batch_idx, (data, targets) in loop:
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate 'runnig' training accurcy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image('minist_images', img_grid)
                writer.add_histogram('fc1', model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )
                writer.add_hparams(
                    {"lr": learning_rate, "bsize": batch_size},
                    {
                        "accuracy": sum(accuracies) / len(accuracies),
                        "loss": sum(losses) / len(losses),
                    },
                )

                if batch_idx == 230:
                    writer.add_embedding(features, metadata=class_labels)
                step += 1

                # update progress bar
                loop.set_description(f"Epoch [{epoch} / {num_epoch}]")
                # loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())
                loop.set_postfix(loss=loss.item())



# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
