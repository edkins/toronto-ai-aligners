import torch
from torchvision.transforms import ToTensor
from torchvision import datasets

def get_train_data(device):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    return _convert(training_data, device)

def get_test_data(device):
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return _convert(test_data, device)

def _convert(data, device):
    X = torch.zeros((len(data), 1, 28, 28), dtype=torch.uint8)
    y = torch.zeros((len(data),), dtype=torch.int64)
    tt = ToTensor()
    for i, (xval,yval) in enumerate(data):
        X[i,:,:,:] = xval * 255
        y[i] = yval
    return X.to(device), y.to(device)
