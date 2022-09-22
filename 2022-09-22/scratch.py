import numpy as np
from torch.nn import Module, Embedding, Sequential, Linear, Flatten, Conv1d, ReLU, GELU, Unflatten, Softmax, CrossEntropyLoss, ModuleList, NLLLoss, Dropout, LogSoftmax
from torch.nn.parameter import Parameter
import torch
import random
import math
import itertools
import sys
import time
from collections import defaultdict

vocab_size = 16384
window_size = 32
embedding_size = 32
output_embedding_size = 32

class MyTransformer(Module):
    def __init__(self, embedding=None, deembedding=None):
        super().__init__()
        if embedding is None:
            e = None
        else:
            e = torch.tensor(embedding.astype('float32'))

        if deembedding is None:
            d = None
        else:
            d = torch.tensor(deembedding.astype('float32'))

        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, _weight=e)
        self.stack = Sequential(GELU(), Linear(2 * embedding_size, output_embedding_size), GELU())
        self.deembed = Linear(output_embedding_size, vocab_size)
        if d is not None:
            self.deembed.weight.data[:] = d.T
            self.deembed.bias.data[:] = 0

    def forward(self, x):
        x = self.embed(x)
        x = self.stack(x)
        #x = self.deembed(x)
        x = x.matmul(self.embed.weight.T)
        return x


def guess_embedding(tokens):
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import coo_array

    print('Calculating sparse matrix')
    values = np.ones((len(tokens)-1))
    #histogram = np.histogram(tokens, bins=vocab_size, range=(0, vocab_size-1))[0].astype('float32') ** -0.5
    #tfreqs = histogram[tokens]
    #values = tfreqs[:-1] * tfreqs[1:]
    coo = coo_array((values, (tokens[:-1], tokens[1:])), shape=(vocab_size,vocab_size))
    print('Calculating csr')
    csr = coo.tocsr()
    print('Calculating embedding')
    embedding = TruncatedSVD(embedding_size).fit_transform(csr)
    print('Done embedding')
    #mean = np.mean(embedding, axis=0).reshape(1, embedding_size)
    var = np.var(embedding)
    result = embedding / np.sqrt(var)
    print(var)
    return result

def guess_embeddings_nmf(tokens):
    from sklearn.decomposition import NMF, TruncatedSVD
    from scipy.sparse import coo_matrix

    print('Calculating sparse matrix')
    values = np.ones((len(tokens)-1))
    #histogram = np.histogram(tokens, bins=vocab_size, range=(0, vocab_size-1))[0].astype('float32') ** -0.5
    #tfreqs = histogram[tokens]
    #values = tfreqs[:-1] * tfreqs[1:]
    coo = coo_matrix((values, (tokens[:-1], tokens[1:])), shape=(vocab_size,vocab_size))
    #coo = coo_matrix((values, (tokens[1:], tokens[:-1])), shape=(vocab_size,vocab_size))
    print('Calculating csr')
    csr = coo.tocsr().log1p()
    print('Calculating embedding')
    nmf = NMF(embedding_size, verbose=2, max_iter=50)
    #nmf = TruncatedSVD(embedding_size)
    embedding = nmf.fit_transform(csr)
    print('Done embedding')
    #mean = np.mean(embedding, axis=0).reshape(1, embedding_size)
    var = np.var(embedding)
    result = embedding #/ np.sqrt(var)
    var2 = np.var(nmf.components_)
    return result, nmf.components_ #/ np.sqrt(var2)


def main():
    tokens = np.fromfile('data/tokens_enwik9', dtype='uint16')
    #tokens = np.fromfile('data/stripped_enwik9.txt', dtype='uint8')
    vocab = []
    for i in range(256):
        vocab.append(i.to_bytes(1, byteorder='little'))

    stop_word = None
    with open('data/vocab_enwik9.txt') as f:
        vocab_text = f.read()
        for i,word in enumerate(vocab_text.split('\n')):
            vocab.append(word.encode('utf-8'))
            if word == '########':
                stop_word = len(vocab)
            if i > vocab_size:
                break
    while len(vocab) < vocab_size:
        vocab.append(b"[INVALID]")

    histogram = np.histogram(tokens, bins=vocab_size, range=(0, vocab_size-1))[0]
    #show_histogram(vocab,histogram)
    weights = (1 + histogram.astype('float32')) ** -0.5
    weights[0] /= 1000000     # training process introduces zeros
    #weights[0] += 3000000    # training process introduces zeros
    #weights = weights.sum() / weights
    print('Calculated weights')

    device = torch.device('cuda')
    mlp = False
    if mlp:
        model = MyMLP().to(device)
    else:
        #e, d = guess_embeddings_nmf(tokens)
        #model = MyTransformer(embedding=e, deembedding=d).to(device)
        model = MyTransformer().to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total = 0
    for p in model.parameters():
        total += np.product(p.shape)
    print(f'{total} parameters')

    chattiness = 1000
    avg = torch.tensor(0.0).to(device)
    switched = False
    start_time = time.monotonic()
    try:
        for i in range(1000000):
            index = random.randrange(0, len(tokens) - window_size - 1)
            snippet = tokens[index:index+window_size+1]
            #for j in range(len(snippet)):
            #    if snippet[j] == stop_word:
            #        snippet = snippet[:j]
            #        break

            if len(snippet) >= 2:
                if mlp:
                    model.train()
                    X = np.zeros((window_size,), dtype='int32')
                    X[1-len(snippet):] = snippet[:-1]
                    y = snippet[-1].astype('int64')
                    X = torch.tensor(X).to(device)
                    y = torch.tensor(y).to(device)
                    pred = model(X)
                else:
                    model.train()
                    X = np.zeros((window_size,), dtype='int32')
                    y = np.zeros((window_size,), dtype='int64')
                    length = min(len(snippet)-1,window_size)
                    X[:length] = snippet[:length]
                    y[:length] = snippet[1:length+1]
                    X = torch.tensor(X).to(device)
                    y = torch.tensor(y).to(device)
                    pred = model(X)
                    #print(X.shape, y.shape, pred.shape)
                #print(vocab[pred.argmax().item()], vocab[y.item()])
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg += loss.detach()
                if i > 0 and i % chattiness == 0:
                    #print(X)
                    #print(y)
                    output = bytearray()
                    for p in X.to('cpu').numpy():
                        output += vocab[p]
                    print(bytes(output))
                    sortable = list(enumerate(pred.to('cpu').detach().numpy()[window_size-1]))
                    sortable.sort(key=lambda pair:pair[1], reverse=True)
                    for t,prob in sortable[:20]:
                        print('    ', vocab[t],prob)
                    #output = bytearray()
                    #for p in pred.argmax(dim=1):
                    #    output += vocab[p]
                    #output2 = bytearray()
                    #for ps in pred.to('cpu').detach().numpy():
                    #    eps = np.exp(ps)
                    #    eps = eps / eps.sum()
                    #    p = random.choices(range(vocab_size), weights=eps, k=1)[0]
                    #    output2 += vocab[p]
                    #print(bytes(output2), pred[window_size-1].max().item())
                    #print(pred[0,32].item())
                    #print(pred[0,0].item())
            if i > 0 and i % chattiness == 0:
                avg_loss = avg.item() / chattiness
                print(i, avg_loss)
                avg = torch.tensor(0.0).to(device)
                if avg_loss < 4.7 and not switched:
                    print('Switching optimizer')
                    switched = True
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    finally:
        end_time = time.monotonic()
        print(f'Time taken: {end_time - start_time}')
        torch.save(model.state_dict(), 'data/model.pth')
        print("Saved model")
        window = np.zeros((window_size,), dtype='int32')
        #window = (np.random.random_sample((window_size,)) * 1000 + 256).astype('int32')
        model.eval()
        output = bytearray()
        with torch.no_grad():
            if mlp:
                window[-1] = 256
                for i in range(100):
                    pred = model(torch.tensor(window).to(device)).argmax().item()
                    output += vocab[pred]
                    window[:-1] = window[1:]
                    window[-1] = pred
            else:
                window[0] = 256
                nxt = 0 #window_size-1
                for i in range(100):
                    ys = model(torch.tensor(window).to(device))
                    #print(window)
                    #print(ys.argmax(dim=1))
                    #print(ys[nxt])
                    eps = np.exp(ys[nxt].to('cpu').detach().numpy())
                    pred = random.choices(range(vocab_size), weights=eps, k=1)[0]
                    #pred = ys[nxt].argmax().item()
                    output += vocab[pred]
                    if nxt == window_size - 1:
                        window[:-1] = window[1:]
                    else:
                        nxt += 1
                    window[nxt] = pred
                    
        print(output)

def load_vocab():
    vocab = []
    for i in range(256):
        vocab.append(i.to_bytes(1, byteorder='little'))
    with open('data/vocab_enwik9.txt') as f:
        vocab_text = f.read()
        for i,word in enumerate(vocab_text.split('\n')):
            vocab.append(word.encode('utf-8'))
            if i > vocab_size:
                break
    while len(vocab) < vocab_size:
        vocab.append(b"[INVALID]")
    return vocab

def predict():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    prompt = input("Enter prompt: ")
    vocab = load_vocab()

    remainder = prompt.encode('utf-8')
    tokens = []
    while len(remainder) > 0:
        longest = ' '
        index = 0
        for i,word in enumerate(vocab):
            if remainder.startswith(word) and len(word) >= len(longest):
                longest = word
                index = i
        remainder = remainder[len(longest):]
        tokens.append(index)
    print(tokens)
    x = np.zeros((window_size,), dtype='int32')
    output = bytearray()
    for i in range(100):
        if len(tokens) < window_size:
            x[:len(tokens)] = tokens 
            pos = len(tokens)-1
        else:
            x[:] = tokens[-window_size:]
            pos = window_size-1
        y = model(torch.tensor(x)).detach().numpy()
        next_token = random.choices(range(vocab_size), weights=np.exp(y[pos]), k=1)[0]
        tokens.append(next_token)
        output += vocab[next_token]
    print(bytes(output))

def analyze():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    X = model.embed.weight.detach().numpy()
    show_embed(X, 5)

def analyze_pca():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    X = model.embed.weight.detach().numpy()
    show_embed(X, reduction='pca')

def analyze1():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    X = model.deembed.weight.detach().numpy()
    show_embed(X, 5)

def analyze_guess(reduction='tsne'):
    tokens = np.fromfile('data/tokens_enwik9', dtype='uint16')
    embedding = guess_embedding(tokens)
    show_embed(embedding, 30, reduction=reduction)

def analyze_guess2(reduction='tsne'):
    tokens = np.fromfile('data/tokens_enwik9', dtype='uint16')
    embedding, _ = guess_embeddings_nmf(tokens)
    show_embed(embedding, 30, reduction=reduction)

def show_embed(X, perplexity=30, reduction='tsne'):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    vocab = load_vocab();
    if reduction == 'tsne':
        reduc = TSNE(verbose=2, perplexity=perplexity)
    else:
        reduc = PCA(2)
    points = reduc.fit_transform(X)
    doc = ['<?xml version="1.0"?>\n','<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">\n']
    size = max(points[:,0].max(), points[:,1].max(), -points[:,0].min(), -points[:,1].min())
    for i,(x,y) in enumerate(points):
        px = 500 + 500 * x / size
        py = 500 + 500 * y / size
        if len(vocab[i]) > 1 or (32 <= vocab[i][0] <= 126 and vocab[i] not in [b'"', b'&', b'<', b'>']):
            doc.append(f'<text font-size="2px" x="{px}" y="{py}">{vocab[i].decode("utf-8")}</text>\n')
    doc.append('</svg>')
    with open('data/graph.svg','w') as f:
        f.write(''.join(doc))

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'eval':
        predict()
    elif sys.argv[1] == 'analyze':
        analyze()
    elif sys.argv[1] == 'analyze_pca':
        analyze_pca()
    elif sys.argv[1] == 'analyze1':
        analyze1()
    elif sys.argv[1] == 'analyze_guess':
        analyze_guess('tsne')
    elif sys.argv[1] == 'analyze_guess_pca':
        analyze_guess('pca')
    elif sys.argv[1] == 'analyze_guess2':
        analyze_guess2('tsne')
    elif sys.argv[1] == 'analyze_guess2_pca':
        analyze_guess2('pca')
