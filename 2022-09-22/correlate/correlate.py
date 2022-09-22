import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from model import MyModel
from data import get_train_data

print('Loading data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)
model.load_state_dict(torch.load('data/model.pth'))
data = get_train_data(device)[0]

size = len(data)
ncount = 28*28

print('Getting activations')
activations = np.zeros((ncount, size),dtype='float32')
for i in range(0, size, 64):
    a = model.get_activations(data[i:i+64], 3).T.detach().cpu()
    activations[:, i:i+64] = a

def attempt2():
    coef = np.corrcoef(activations)
    print(coef)
    reduce = TSNE(2, verbose=2, perplexity=30, metric='precomputed')
    embed = reduce.fit_transform(1 - coef * coef)
    plt.scatter(embed[:,0], embed[:,1])
    plt.show()

def attempt1():
    norms = np.sqrt((activations * activations).sum(axis=1))
    normed = activations / norms.reshape((ncount,1))
    reduce = TSNE(2, verbose=2, perplexity=30)
    embed = reduce.fit_transform(normed * normed)

    plt.scatter(embed[:,0], embed[:,1])
    plt.show()

attempt2()
