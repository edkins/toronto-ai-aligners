import numpy as np
from torch.nn import Module, Embedding, Sequential, Linear, Flatten, Conv1d, ReLU, Unflatten, Softmax, CrossEntropyLoss, ModuleList, NLLLoss, Dropout, LogSoftmax, L1Loss
from torch.nn.parameter import Parameter
import torch
import random
import math
import itertools
import time
import sys

import imaging

window_size = 32
embedding_size = 112
index_size = 16
vocab_size = 4096
n_heads = 2
n_layers = 4
t_key_size = 32
t_value_size = 64
batch_size = 64

class AttentionHead(Module):
    def __init__(self, in_size, value_size, key_size):
        super().__init__()
        self.wk = Linear(in_size, key_size)
        self.wv = Linear(in_size, value_size)
        self.wq = Linear(in_size, key_size, bias=False)
        tri = (np.tril(np.full((window_size, window_size),2.0, dtype='float32')) - 1) * 100000.0
        self.tri = torch.tensor(tri.reshape((1, window_size, window_size))).to(torch.device('cuda'))
        self.scale = t_key_size ** -0.5

    def forward(self, x):
        k = self.wk(x)
        q = self.wq(x)
        v = self.wv(x)
        a = (k.matmul(q.transpose(1,2)) * self.scale).minimum(self.tri).softmax(dim=2)
        result = a.matmul(v)
        return result

class TransformerLayer(Module):
    def __init__(self, in_out_size):
        super().__init__()
        self.heads = ModuleList([AttentionHead(in_out_size, t_value_size, t_key_size) for _ in range(n_heads)])
        self.dropout = Dropout(p=0.1)
        self.linear = Linear(t_value_size * n_heads, in_out_size)
        self.linear2 = Linear(in_out_size, in_out_size)
        self.relu = ReLU()
        self.linear3 = Linear(in_out_size, in_out_size)
        self.dropout2 = Dropout(p=0.1)
        self.in_out_size = in_out_size

    def forward(self, x):
        y = torch.cat([self.heads[i](x) for i in range(n_heads)], dim=2)
        y = self.linear(y)
        y = self.dropout(y)
        y = torch.nn.functional.layer_norm(x + y, (self.in_out_size,))
        z = self.linear2(y)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.dropout(z)
        return torch.nn.functional.layer_norm(y + z, (self.in_out_size,))

class MyTransformer(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(num_embeddings=vocab_size + 2, embedding_dim=embedding_size-2)
        self.index = Parameter(torch.rand((1, window_size, index_size)))
        self.dropout = Dropout(p=0.1)
        layers = []
        in_size = embedding_size + index_size
        for i in range(n_layers):
            layers.append(TransformerLayer(in_size))
        self.layers = Sequential(*layers)
        self.deembed = Linear(in_size, 4 * vocab_size)

    def forward(self, x):
        space = (x & 1).reshape(x.shape[0], x.shape[1], 1)
        capital = ((x & 2) // 2).reshape(x.shape[0], x.shape[1], 1)
        x = torch.cat([self.embed(x//4), space, capital, self.index.tile(x.shape[0], 1, 1)], dim=2)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.deembed(x)
        return x


def main():
    tokens = np.fromfile('data/tokens_enwik9', dtype='uint16')
    #tokens = np.fromfile('data/stripped_enwik9.txt', dtype='uint8')
    vocab = load_vocab()

    #weights = 1 + np.histogram(tokens, bins=vocab_size, range=(0, vocab_size-1))[0].astype('float32')
    #weights[0] += 30000000    # training process introduces zeros
    #weights = weights.sum() / weights
    #print('Calculated weights')

    device = torch.device('cuda')
    model = MyTransformer().to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=1e-5)
    opt_version = 0
    total = 0
    for p in model.parameters():
        total += np.product(p.shape)
    print(f'{total} parameters')

    imager = imaging.Imager('data/params.png', segments=imaging.choose_segments(model, 256))

    chattiness = 200
    avg = torch.tensor(0.0).to(device)
    tokens_seen  = 0
    start_time = time.monotonic()
    try:
        for i in range(1000000):
            #if opt_version == 0 and tokens_seen > 2_000_000:
            #    print('Switching optimizer')
            #    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
            #    opt_version = 1
            #if opt_version == 1 and tokens_seen > 10_000_000:
            #    print('Switching optimizer again')
            #    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            #    opt_version = 2
            index = random.randrange(0, len(tokens) - window_size - 1)
            Xs = []
            ys = []
            while len(Xs) < batch_size:
                snippet = tokens[index:index+window_size+1]
                X = np.zeros((window_size,), dtype='int32')
                y = np.zeros((window_size,), dtype='int64')
                length = min(len(snippet)-1,window_size)
                X[:length] = snippet[:length]
                y[:length] = snippet[1:length+1]
                Xs.append(X)
                ys.append(y)
                tokens_seen += length

            model.train()
            X = torch.tensor(np.array(Xs)).to(device)
            y = torch.tensor(np.array(ys)).to(device)
            pred = model(X)
            loss = loss_fn(pred.reshape(batch_size * window_size, 4 * vocab_size), y.reshape((batch_size * window_size,)))
            optimizer.zero_grad()
            loss.backward()
            if i > 0 and i % chattiness == 0:
                sd = model.state_dict()
                params = model.parameters()
                for param,val in zip(sd, params):
                    if val.grad is not None and len(val.shape) == 1:
                        print(param, torch.linalg.vector_norm(val.grad).item())
                        #print(val.grad)
                    elif val.grad is not None and len(val.shape) == 2:
                        print(param, torch.linalg.matrix_norm(val.grad).item())
                        #print(val.grad)
                    else:
                        print(param)
            optimizer.step()
            avg += loss.detach()
            if i % 64 == 0:
                imager.extend_from_model(model)
            if i > 0 and i % chattiness == 0:
                output = bytearray()
                for p in X.to('cpu').numpy()[0]:
                    output += vocab[p]
                print(bytes(output))
                preds = pred[0].to('cpu').detach().numpy()[window_size-1]
                sortable = list(enumerate(preds))
                sortable.sort(key=lambda pair:pair[1], reverse=True)
                for t,prob in sortable[:20]:
                    print('    ', vocab[t],prob)

                print(i, f'{tokens_seen/1000000}m tokens seen', avg.item() / chattiness, f'     {int(time.monotonic()-start_time)} seconds')
                avg *= 0

                imager.save()

    finally:
        end_time = time.monotonic()
        print(f'Time taken: {end_time - start_time}. {tokens_seen/(end_time-start_time)} tokens/second')
        torch.save(model.state_dict(), 'data/model.pth')
        print("Saved model")
        window = np.zeros((window_size,), dtype='int32')
        #window = (np.random.random_sample((window_size,)) * 1000 + 256).astype('int32')
        model.eval()
        output = bytearray()
        with torch.no_grad():
            window[0] = 256
            nxt = 0 #window_size-1
            for i in range(100):
                ys = model(torch.tensor(window.reshape(1,window_size)).to(device))[0][nxt]
                #print(window)
                #print(ys.argmax(dim=1))
                #print(ys[nxt])
                space = random.random() < ys[vocab_size]
                capital = random.random() < ys[vocab_size+1]
                eps = np.exp(ys.to('cpu').detach().numpy())
                pred = random.choices(range(4*vocab_size), weights=eps, k=1)[0]
                #pred = ys[nxt].argmax().item()
                output += vocab[pred]
                if nxt == window_size - 1:
                    window[:-1] = window[1:]
                else:
                    nxt += 1
                window[nxt] = pred
                    
        print(output)

        inp = (np.random.random_sample((1,window_size)) * vocab_size).astype('int32')
        inp2 = np.array(inp)
        inp2[0,window_size // 2] += 1
        outp = model(torch.tensor(inp).to(device)).detach().cpu().numpy()
        outp2 = model(torch.tensor(inp2).to(device)).detach().cpu().numpy()
        print(inp)
        print(inp2)
        print(outp[0,:,0])
        print(outp2[0,:,0])

def append_vocab(vocab, word):
    vocab.append(word)
    vocab.append(b' ' + word)
    if word[0] >= 97 and word[0] <= 122:
        capital = (word[0] - 32).to_bytes(1, byteorder='little') + word[1:]
    else:
        capital = word
    vocab.append(capital)
    vocab.append(b' ' + capital)

def load_vocab():
    vocab = []
    for i in range(256):
        append_vocab(vocab, i.to_bytes(1, byteorder='little'))
    with open('data/vocab_enwik9.txt') as f:
        vocab_text = f.read()
        for i,word in enumerate(vocab_text.split('\n')):
            append_vocab(vocab, word.encode('utf-8'))
            if i > vocab_size:
                break
    while len(vocab) < vocab_size:
        vocab.append(b"[INVALID]")
    return vocab

def predict():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    temperature = 0.65 #float(input("Temperature: ") or '1')
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
    x = np.zeros((batch_size, window_size), dtype='int32')
    output = bytearray()
    for i in range(100):
        if len(tokens) < window_size:
            x[0,:len(tokens)] = tokens 
            pos = len(tokens)-1
        else:
            x[0,:] = tokens[-window_size:]
            pos = window_size-1
        y = model(torch.tensor(x)).detach().numpy()
        next_token = random.choices(range(4*vocab_size), weights=np.exp(y[0,pos] / temperature), k=1)[0]
        tokens.append(next_token)
        output += vocab[next_token]
    print(bytes(output))

def analyze():
    model = MyTransformer()
    model.load_state_dict(torch.load('data/model.pth'))
    x0 = model.embed.weight.detach().numpy()[:vocab_size,:]
    print(x0.shape)
    xk = model.layers[0].heads[0].wq.weight.detach().numpy()[:,:embedding_size-2]
    xq = model.layers[0].heads[0].wq.weight.detach().numpy()[:,:embedding_size-2]
    #show_embed(np.matmul(x0,xk.T), 30)
    xk1 = np.matmul(x0,xk.T)
    xq1 = np.matmul(x0,xq.T)
    xk2 = xk1 / np.linalg.norm(xk1, axis=1).reshape(vocab_size, 1) 
    xq2 = xq1 / np.linalg.norm(xq1, axis=1).reshape(vocab_size, 1) 
    dots = np.matmul(xk2, xq2.T)
    matrix = np.arccos(np.minimum(1, np.maximum(-1, np.maximum(dots, dots.T))))
    print(matrix.min(), matrix.max())
    show_embed(matrix, 30, 'precomputed')

def show_embed(X, perplexity=30, reduction='tsne'):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    vocab = load_vocab();
    if reduction == 'tsne':
        reduc = TSNE(verbose=2, perplexity=perplexity)
    elif reduction == 'precomputed':
        reduc = TSNE(verbose=2, perplexity=perplexity, metric='precomputed')
    else:
        reduc = PCA(2)

    if X.shape[1] == 2:
        points = X
    else:
        points = reduc.fit_transform(X)
    doc = ['<?xml version="1.0"?>\n','<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">\n']
    size = max(abs(points[:,0].max()), abs(points[:,1].max()), abs(points[:,0].min()), abs(points[:,1].min()))
    for i,(x,y) in enumerate(points):
        px = 500 + 500 * x / size
        py = 500 + 500 * y / size
        j = 4 * i
        tsize = 100 * ((i + 256) ** -0.5)
        if len(vocab[j]) > 1 or (32 <= vocab[j][0] <= 126 and vocab[j] not in [b'"', b'&', b'<', b'>']):
            try:
                doc.append(f'<text font-size="{tsize}px" x="{px}" y="{py}">{vocab[j].decode("utf-8")}</text>\n')
            except:
                pass
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
