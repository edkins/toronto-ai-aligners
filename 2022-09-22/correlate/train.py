import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os

from model import MyModel
from data import get_train_data, get_test_data

def train(data, model, loss_fn, optimizer, batch_size):
    size = len(data[0])
    indices = np.arange(0, size)
    np.random.shuffle(indices)

    model.train()
    for i in range(0, size, batch_size):
        chunk = indices[i:i+64]
        X = data[0][chunk]
        y = data[1][chunk]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i//batch_size) % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{i:>5d}/{size:>5d}]")

def test(data, model, loss_fn, batch_size, device):
    size = len(data[0])
    model.eval()
    test_loss = torch.tensor(0.0).to(device)
    correct = torch.tensor(0).to(device)
    num_batches = 0
    with torch.no_grad():
        for i in range(0, size, batch_size):
            X = data[0][i:i+batch_size]
            y = data[1][i:i+batch_size]
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).sum()
            num_batches += 1
    print(f"Test Error: \n Accuracy: {(100*correct.item()/size):>0.1f}%, Avg loss: {test_loss.item()/num_batches:>8f} \n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

train_data = get_train_data(device)
test_data = get_test_data(device)

batch_size = 64

model = MyModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, loss_fn, optimizer, batch_size)
    test(test_data, model, loss_fn, batch_size, device)
print("Done!")

os.makedirs('data', exist_ok=True)
torch.save(model.state_dict(), 'data/model.pth')
