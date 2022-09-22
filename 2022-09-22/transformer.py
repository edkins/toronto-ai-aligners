import numpy as np
from torch.nn import Module, Embedding, Sequential, Linear, Flatten, Conv1d, ReLU, Unflatten, Softmax, CrossEntropyLoss, ModuleList, NLLLoss, Dropout, LogSoftmax
from torch.nn.parameter import Parameter
import torch
import random
import math
import itertools
import sys
from collections import defaultdict

window_size = 32
embedding_size = 120
index_size = 8
output_embedding_size = embedding_size
mlp_hidden_size = 128
mlp_layers = 4
vocab_size = 16384
n_heads = 4
n_layers = 2
t_key_size = 32
t_value_size = 32

class AttentionHead(Module):
    def __init__(self, in_size, value_size, key_size):
        super().__init__()
        self.wk = Parameter(torch.rand((in_size, key_size)))
        self.wv = Parameter(torch.rand((in_size, value_size)))
        self.wq = Parameter(torch.rand((in_size, key_size)))
        tri = (np.tril(np.full((window_size, window_size),2.0, dtype='float32')) - 1) * 100000.0
        self.tri = Parameter(torch.tensor(tri), requires_grad=False)
        self.scale = 1 / 256

    def forward(self, x):
        #x = torch.nn.functional.normalize(x)
        k = x.matmul(self.wk)
        q = x.matmul(self.wq)
        v = x.matmul(self.wv)
        a = (q.matmul(k.T) * self.scale).minimum(self.tri).softmax(dim=1)
        #if random.random() < 0.001:
        #    print(a)
        result = a.matmul(v)
        return result

class TransformerLayer(Module):
    def __init__(self, in_out_size):
        super().__init__()
        self.in_out_size = in_out_size
        self.heads = ModuleList([AttentionHead(in_out_size, t_value_size, t_key_size) for _ in range(n_heads)])
        self.dropout = Dropout(p=0.1)
        self.linear = Linear(t_value_size * n_heads, in_out_size)
        self.linear2 = Linear(in_out_size, 512)
        self.relu = ReLU()
        self.linear3 = Linear(512, in_out_size)
        self.dropout2 = Dropout(p=0.1)
        self.bypass = False

    def forward(self, x):
        if self.bypass:
            y = x
        else:
            y = torch.cat([self.heads[i](x) for i in range(n_heads)], dim=1)
            y = self.linear(y)
            y = self.dropout(y)
            #y = torch.nn.functional.layer_norm(x + y, (self.in_out_size,))
            y = torch.nn.functional.layer_norm(x + y, (self.in_out_size,))
        z = self.linear2(y)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.dropout2(z)
        #return torch.nn.functional.normalize(y + z, dim=1)
        return torch.nn.functional.layer_norm(y + z, (self.in_out_size,))
        #return y + z

class MyTransformer(Module):
    def __init__(self, embedding=None):
        super().__init__()
        if embedding is None:
            e = None
        else:
            e = torch.tensor(embedding.astype('float32'))
        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, _weight=e)
        self.index = Parameter(torch.rand((window_size, index_size)))
        self.dropout = Dropout(p=0.1)
        layers = []
        in_size = embedding_size + index_size
        self.bypass = False
        self.bypass_layer = Linear(in_size, in_size)
        for i in range(n_layers):
            layers.append(TransformerLayer(in_size))
            #layers.append(AttentionHead(in_size, in_size, t_key_size))
        self.layers = Sequential(*layers)
        self.deembed = Linear(in_size, vocab_size)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, x):
        #index = torch.eye(window_size).to('cuda')
        x = torch.cat([self.embed(x), self.index], dim=1)
        x = self.dropout(x)
        if self.bypass:
            x = self.bypass_layer(x)
        else:
            x = self.layers(x)
        x = self.deembed(x)
        #x = x[:,:embedding_size]
        #x = x.matmul(self.embed.weight.T)
        x = self.softmax(x)
        return x


class MyMLP(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        mlp = [Flatten(start_dim=0)]
        in_size = embedding_size * window_size
        for i in range(mlp_layers):
            out_size = (output_embedding_size * window_size) if i==mlp_layers-1 else mlp_hidden_size
            mlp.append(Linear(in_size, out_size))
            mlp.append(ReLU())
            in_size = out_size
        #mlp.append(Unflatten(0, (window_size, output_embedding_size)))
        self.mlp = Sequential(*mlp)
        #self.deembed = Conv1D(in_channels=output_embedding_size, out_channels=vocab_size, kernel_size=1)
        #self.softmax = SoftMax(dim=1)
        self.deembed = Linear(window_size * output_embedding_size, vocab_size)
        self.softmax = Softmax(dim=0)

    def forward(self, x):
        x = self.embed(x)
        x = self.mlp(x)
        x = self.deembed(x)
        x = self.softmax(x)
        return x

def show_histogram(vocab,h):
    buckets = defaultdict(list)
    for i,count in enumerate(h):
        word = vocab[i]
        if count > 0:
            bucket = math.floor(math.log10(count))
            buckets[bucket].append(word)
    for i,words in sorted(buckets.items()):
        print(f'10^{i}: ({len(words)}) {words[:100]}')

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
        #model = MyTransformer(embedding=guess_embedding(tokens)).to(device)
        model = MyTransformer(embedding=None).to(device)
    #loss_fn = CrossEntropyLoss(weight=torch.tensor(weights.astype('float32')).to(device))
    #loss_fn = CrossEntropyLoss(weight=None, label_smoothing=0.1)
    #loss_fn = NLLLoss(weight=torch.tensor(weights.astype('float32')).to(device))
    loss_fn = NLLLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=10, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.98))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.98))
    total = 0
    for p in model.parameters():
        total += np.product(p.shape)
    print(f'{total} parameters')

    chattiness = 1000
    avg = torch.tensor(0.0).to(device)
    try:
        for i in range(1000000):
            #if i == 10000:
            #    print('Switching optimizer')
            #    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            index = random.randrange(0, len(tokens) - window_size - 1)
            snippet = tokens[index:index+window_size+1]
            for j in range(len(snippet)):
                if snippet[j] == stop_word:
                    snippet = snippet[:j]
                    break

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
                print(i, avg.item() / chattiness)
                avg *= 0
    finally:
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

def analyze1():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    X = model.deembed.weight.detach().numpy()
    show_embed(X, 5)

def analyze2():
    tokens = np.fromfile('data/tokens_enwik9', dtype='uint16')
    embedding = guess_embedding(tokens)
    show_embed(embedding, 30)

def show_embed(X, perplexity):
    from sklearn.manifold import TSNE
    vocab = load_vocab();
    tsne = TSNE(verbose=2, perplexity=perplexity)
    points = tsne.fit_transform(X)
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
    elif sys.argv[1] == 'analyze1':
        analyze1()
    elif sys.argv[1] == 'analyze_guess':
        analyze2()
