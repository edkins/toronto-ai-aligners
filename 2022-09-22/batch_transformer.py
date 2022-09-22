import numpy as np
from torch.nn import Module, Embedding, Sequential, Linear, Flatten, Conv1d, ReLU, Unflatten, Softmax, CrossEntropyLoss, ModuleList, NLLLoss, Dropout, LogSoftmax
from torch.nn.parameter import Parameter
import torch
import random
import math
import itertools

window_size = 10
embedding_size = 16
index_size = 4
output_embedding_size = embedding_size
mlp_hidden_size = 128
mlp_layers = 4
vocab_size = 16384
n_heads = 4
n_layers = 4
t_key_size = 8
t_value_size = 8

class AttentionHead(Module):
    def __init__(self, in_size, value_size, key_size):
        super().__init__()
        self.wk = Parameter(torch.rand((in_size, key_size)))
        self.wv = Parameter(torch.rand((in_size, value_size)))
        self.wq = Parameter(torch.rand((in_size, key_size)))
        tri = (np.tril(np.full((window_size, window_size),2.0, dtype='float32')) - 1) * 100000.0
        self.tri = Parameter(torch.tensor(tri.reshape(1, window_size, window_size)), requires_grad=False)

    def forward(self, x):
        k = x.matmul(self.wk)
        q = x.matmul(self.wq)
        v = x.matmul(self.wv)
        a = k.matmul(q.transpose(1,2)).minimum(self.tri).softmax(dim=2)
        #a = k.matmul(q.T).softmax(dim=1)
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

    def forward(self, x):
        y = torch.cat([self.heads[i](x) for i in range(n_heads)], dim=2)
        y = self.linear(y)
        y = self.dropout(y)
        y = torch.nn.functional.normalize(x + y, dim=2)
        z = self.linear2(y)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.dropout(z)
        return torch.nn.functional.normalize(y + z, dim=2)

class MyTransformer(Module):
    def __init__(self):
        super().__init__()
        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.index = Parameter(torch.rand((1, window_size, index_size)))
        self.dropout = Dropout(p=0.1)
        layers = []
        in_size = embedding_size + index_size
        for i in range(n_layers):
            layers.append(TransformerLayer(in_size))
            #layers.append(AttentionHead(in_size, in_size, t_key_size))
        self.layers = Sequential(*layers)
        self.deembed = Linear(in_size, vocab_size)
        self.softmax = LogSoftmax(dim=2)

    def forward(self, x):
        x = torch.cat([self.embed(x), self.index.tile(x.shape[0], 1, 1)], dim=2)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.deembed(x)
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
            if i > 65000:
                break
    while len(vocab) < 65536:
        vocab.append(b"[INVALID]")

    weights = 1 + np.histogram(tokens, bins=vocab_size, range=(0, vocab_size-1))[0].astype('float32')
    weights[0] += 30000000    # training process introduces zeros
    weights = weights.sum() / weights
    print('Calculated weights')

    device = torch.device('cuda')
    mlp = False
    if mlp:
        model = MyMLP().to(device)
    else:
        model = MyTransformer().to(device)
    #loss_fn = CrossEntropyLoss(weight=torch.tensor(weights.astype('float32')).to(device))
    #loss_fn = CrossEntropyLoss(weight=None, label_smoothing=0.1)
    #loss_fn = NLLLoss(weight=torch.tensor(weights.astype('float32')).to(device))
    loss_fn = NLLLoss(weight=None)
    #optimizer = torch.optim.SGD(model.parameters(), lr=10, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5, betas=(0.9, 0.98))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.98))
    total = 0
    for p in model.parameters():
        total += np.product(p.shape)
    print(f'{total} parameters')

    chattiness = 200
    avg = torch.tensor(0.0).to(device)
    try:
        for i in range(1000000):
            #if i == 10000:
            #    print('Switching optimizer')
            #    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            index = random.randrange(0, len(tokens) - window_size - 1)
            Xs = []
            ys = []
            while len(Xs) < 64:
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
                        X = np.zeros((window_size,), dtype='int32')
                        y = np.zeros((window_size,), dtype='int64')
                        length = min(len(snippet)-1,window_size)
                        X[:length] = snippet[:length]
                        y[:length] = snippet[1:length+1]
                        Xs.append(X)
                        ys.append(y)

            model.train()
            X = torch.tensor(np.array(Xs)).to(device)
            y = torch.tensor(np.array(ys)).to(device)
            pred = model(X)
            #print(X.shape, y.shape, pred.shape)
            #print(vocab[pred.argmax().item()], vocab[y.item()])
            loss = loss_fn(pred.reshape(64 * window_size, vocab_size), y.reshape(64 * window_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg += loss.detach()
            if i > 0 and i % chattiness == 0:
                output = bytearray()
                for p in X.to('cpu').numpy()[0]:
                    output += vocab[p]
                print(bytes(output))
                sortable = list(enumerate(pred[0].to('cpu').detach().numpy()[window_size-1]))
                sortable.sort(key=lambda pair:pair[1], reverse=True)
                for t,prob in sortable[:20]:
                    print('    ', vocab[t],prob)

                #output2 = bytearray()
                #for ps in pred[0].to('cpu').detach().numpy():
                #    eps = np.exp(ps)
                #    eps = eps / eps.sum()
                #    p = random.choices(range(vocab_size), weights=eps, k=1)[0]
                #    output2 += vocab[p]
                #print(bytes(output2), pred[0][window_size-1].max().item())
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
                    ys = model(torch.tensor(window.reshape(1,window_size)).to(device))[0]
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

        inp = (np.random.random_sample((1,window_size)) * vocab_size).astype('int32')
        inp2 = np.array(inp)
        inp2[0,window_size // 2] += 1
        outp = model(torch.tensor(inp).to(device)).detach().cpu().numpy()
        outp2 = model(torch.tensor(inp2).to(device)).detach().cpu().numpy()
        print(inp)
        print(inp2)
        print(outp[0,:,0])
        print(outp2[0,:,0])

if __name__ == '__main__':
    main()
