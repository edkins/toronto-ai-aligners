with open('original_data/train-images-idx3-ubyte', 'rb') as f:
    data = f.read()

# print(type(data))
# print(len(data))

import torch

training_image_file_data = torch.tensor(list(data[16:]), dtype=torch.uint8)
# print(training_image_file_data.shape)
X = training_image_file_data.reshape(60000, 1, 28, 28)
# print(X.shape)
# print(X.dtype)

with open('original_data/train-labels-idx1-ubyte', 'rb') as f:
    data = f.read()

y = torch.tensor(list(data[8:]), dtype=torch.uint8)
# print(y.shape)
# print(y.dtype)

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.dense0 = torch.nn.Linear(28*28, 100)

    def forward(self, x):
        return self.dense0(self.flatten(x))

model = MyModel()
#print(model(X[:256]))

opt = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(100):
    for j in range(0, 60000, 64):
        loss = loss_fn(model(X[j:j+64].to(torch.float32)), y[j:j+64])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(loss)
