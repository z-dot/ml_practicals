# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
dataset_test = datasets.MNIST("./mnist", train=False, transform=transform)

# %%

fig = plt.figure()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(dataset_train[i][0].reshape(28, 28), cmap="Greys_r")
# %%


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                1, 25, kernel_size=(12, 12), stride=(2, 2), padding=0
            ),  # we can calculate output size as 1 + (input_size - kernel_size) / stride
            # so we're at 9x9
            nn.ReLU(),
            nn.Conv2d(25, 64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            # now still at 9x9, padding chosen to stay at 9x9
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                1024, 10
            ),  # maybe another nonlinearity after this? edit: don't think so
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# %%

model = MNISTCNN()
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainset = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
validset = torch.utils.data.DataLoader(dataset_valid, batch_size=64, shuffle=False)
testset = torch.utils.data.DataLoader(dataset_test, batch_size=64)

num_iterations = 5000

iteration = 0
while iteration < num_iterations:
    model.train()
    for images, labels in trainset:
        if iteration >= num_iterations:
            break
        optimizer.zero_grad()
        outputs = model(images)
        loss = ce_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        iteration += 1

        if iteration % 100 == 0:
            model.eval()
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for images, labels in validset:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()

            print(f"iteration {iteration}\nvalid: {100 * valid_correct / valid_total}%")

        if iteration % 500 == 0:
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for images, labels in trainset:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

            print(f"train: {100 * train_correct / train_total}%")
# %%
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in testset:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f"final accuracy: {100 * test_correct / test_total}%")

# %%
