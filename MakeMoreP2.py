import torch
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# print(torch.cuda.is_available())       # True
# print(torch.version.cuda)              # Should print 12.1
# print(torch.cuda.get_device_name(0))

words = open('../MakeMore-Backpropagation/Names_Farsi.txt', 'r', encoding= 'utf-8').read().splitlines()

chars = sorted(list(set(''.join(words))))
vocab_size = len(chars)+1
stoi = {S: i + 1 for i, S in enumerate(chars)}
stoi['.'] = 0
itos = {i : S for S , i in stoi.items()}

block_size = 3
X, Y = [] , []

for word in words:
    context = [0] * block_size
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '------>', itos[ix])
        context = context[1:] + [ix]
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
# print(X.shape)
# g = torch.Generator().manual_seed(2147483647)
# C = torch.randn((vocab_size , 10 ), generator=g)

# F.one_hot(torch.tensor(5) , num_classes= vocab_size).float() @ C

def build_dataset (words):
    block_size = 3
    X, Y = [] , []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '------>', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    return X, Y

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

print(f'{len(words)} length of all the words in our data set')

print(f'{len(Xtr)} training examples , {len(Xdev)} validation examples , {len(Xtest)} test examples')
#Hypremeters
n_emb = 20
n_hidden = 300
dropout_rate = 0.3
weight_decay = 0.01
max_steps = 200000
batch_size = 32
initial_lr = 0.1

g = torch.Generator(device=device).manual_seed(2147483647)
if device.type == 'cuda':
    C = torch.randn((vocab_size, n_emb), device=device)
else:
    C = torch.randn((vocab_size , n_emb), generator=g, device=device)
if device.type == 'cuda':
        W1 = torch.randn((n_emb * block_size, n_hidden), device=device) * (5 / 3) / ((n_emb * block_size) ** 0.5)
else:
    W1 = torch.randn((n_emb * block_size , n_hidden ), generator = g, device=device) * (5/3) / ((n_emb * block_size) ** 0.5) #-- Calculated the initialized scale by applying Gain / (Fan_n) ** 0.5 NOTE: Fan_n is the amount of the inputs to our neural network
#b1 = torch.randn(n_hidden , generator = g) *0.001

# rs = torch.cat([emb[: , 0,:], emb[: , 1, :], emb[: , 2, :]] , 1)
# torch.cat(torch.unbind(emb, 1),1)
# print(emb.view(15,6))
# print(h.shape)
if device.type == 'cuda':
    W2 = torch.randn((n_hidden, vocab_size), device=device) * 0.01
else:
    W2 = torch.randn((n_hidden, vocab_size) , generator = g, device=device) * 0.01

if device.type == 'cuda':
    b2 = torch.rand(vocab_size, device=device) * 0
else:
    b2 = torch.rand(vocab_size , generator = g, device=device) * 0

bngain = torch.ones((1 , n_hidden), device=device)
bnbias  = torch.zeros((1 , n_hidden), device=device)
bnmean_running = torch.zeros((1 , n_hidden), device=device)
bnstd_running = torch.ones((1 , n_hidden), device=device)
parameters = [C, W1, W2, b2 , bngain, bnbias]

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3 , 0 , 1000)
lrs = 10 ** lre

lri = []
lossi = []
iStep = []

#-- this is for training the data on the whole dataset
# for step in range(10000):
#  #-- mini batch
#  ix = torch.randint(0, X.shape[0], (40,))
#  #-- Forward Pass
#  emb = C[X[ix]]
#  h = torch.tanh(emb.view(-1 , 6) @ W1 + b1)
#  logits = h @ W2 + b2
#  # print(logits.shape)
#
# # counts = logits.exp()
# # prob = counts / counts.sum(1 , keepdims = True)
# # # print(prob[1].sum())
# # loss = -prob[torch.arange(vocab_size),Y].log().mean()
#  loss = F.cross_entropy(logits , Y[ix])
#  # print(loss)
#
# #-- Backward Pass
#  for p in parameters:
#     p.grad = None
#  loss.backward()
#
#  # lr = lrs[step]
#  lr = 0.618783
#  for p in parameters:
#    p.data += -lr * p.grad

 #-- track stats
 # lri.append(lre[step])
 # lossi.append(loss.item())

 # print(f'{step} = {loss.item()}')

# plt.plot(lri, lossi)
# plt.xlabel('step')
# plt.ylabel('loss')
# # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# plt.show()

for step in range(max_steps):
 #-- mini batch
 if device.type == 'cuda':
     ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=device)
 else:
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g, device=device)
 Xb , Yb = Xtr[ix], Ytr[ix] # Batch X,Y
 #-- Forward Pass
 emb = C[Xb]
 embcat = emb.view(emb.shape[0] , -1)
 hpreact = embcat @ W1 #+ b1 this bias will be useless because in hpreact layer we subtract it and bnbias is actually in charge of bias right now
 # we define the mean and standard deviation like this because in the previous version it always required a batch as an input but this way we can input a single input
 #-- BatchNorm Layer
 # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 bnmeani = hpreact.mean(0 , keepdim = True)
 bnstdi = hpreact.std(0 , keepdim = True)
 hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
 # hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
 # hpreact = emb.view(-1 , n_emb * block_size) @ W1 + b1
 with torch.no_grad():
     bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
     bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 #-- Non Linearity
 h = torch.tanh(hpreact)
 # new line that I got from chatgpt to prevent the model from overfiting {
 h = F.dropout(h, p=0.2, training=True)
 # }
 logits = h @ W2 + b2
 # print(logits.shape)

# counts = logits.exp()
# prob = counts / counts.sum(1 , keepdims = True)
# # print(prob[1].sum())
# loss = -prob[torch.arange(vocab_size),Y].log().mean()
 loss = F.cross_entropy(logits , Yb)
 # print(loss)

#-- Backward Pass
 for p in parameters:
    p.grad = None
 loss.backward()

 # lr = lrs[step]
 lr = 0.1 if step < (max_steps/2) else 0.01

 for p in parameters:

   p.data += -lr * (p.grad + 0.001 * p.data)

 # Tracking stats
 if step % 10000 == 0:
     print(f'{step:7d}/{max_steps:7d}, loss: {loss.item():.8f}')

 break


#-- matplot
 lossi.append(loss.log10().item())
 iStep.append(step)

# print(loss.item())
# plt.plot(iStep , lossi)
# plt.show()

# emb = C[Xdev]
# h = torch.tanh(emb.view(-1 , n_emb * block_size) @ W1 + b1)
# logits = h @ W2 + b2
# loss = F.cross_entropy(logits , Ydev)
# print(loss.item())

# plt.figure(figsize=(8, 8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s = 200)
# for i in range(C.shape[0]):
#     plt.text(C[i ,0].item(), C[i ,1].item(), itos[i] , ha='center', va='bottom', color = 'white')
# plt.grid('minor')
# plt.show()

@torch.no_grad()
def split_loss(split):
    x,y ={
        'train' : (Xtr , Ytr),
        'val' : (Xdev , Ydev),
        'test' : (Xtest , Ytest),
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0] , -1)
    hpreact = embcat @ W1 #+ b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits , y)
    print(split ,loss.item())

split_loss('train')
split_loss('val')
#-- sampling from the model
# g = torch.Generator().manual_seed(2147483647 + 10)
#
# for _ in range(10):
#     out = []
#     context = [0] * block_size
#     while True:
#         #forward pass
#         emb = C[torch.tensor([context])]
#         embcat = emb.view(emb.shape[0] , -1)
#         hpreact = embcat @ W1 #+ b1
#         bnmeani = hpreact.mean(0, keepdim=True)
#         bnstdi = hpreact.std(0, keepdim=True) + 100
#         hpreact = bngain * (hpreact - bnmeani ) / bnstdi + bnbias
#         h = torch.tanh(hpreact)
#         logits = h @ W2 + b2
#         probs = F.softmax(logits , dim=1)
#         #sample from the distribution
#         ix = torch.multinomial(probs , 1 , generator=g).item()
#         context = context[1:] + [ix]
#         out.append(ix)
#         #if we sample special '.' token, break
#         if ix == 0:
#             break
#     print(''.join(itos[i] for i in out))#decode and print the generated word


# arr = (h.abs().detach().numpy() > 0.99)
#
# arr_grid = arr[:300].reshape(30, 10)
#
# plt.imshow(arr_grid, cmap="gray", interpolation="nearest")
# plt.show()

#-- Training the network with more PyTorch like approach

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, use_normal = False, gain=1.0, device='cpu'):
        # self.weight = torch.randn((fan_in, fan_out), generator = g) / fan_in**0.5
        # self.bias = torch.zeros(fan_out) if bias else None
        if use_normal:
            std = gain * (2.0 / (fan_in + fan_out))** 0.5
            self.weight = torch.randn((fan_in , fan_out), generator = g, device=device) * std

        else:
            a = gain * (6.0 / (fan_in + fan_out))** 0.5
            self.weight = (torch.rand((fan_in , fan_out), generator = g, device=device)* 2 - 1) * a

        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters (self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BenchNorm:
    def __init__(self, dim, eps = 1e-5 ,momentum = 0.1 , device='cpu'):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.device = device
        #parameters (trained with backprop)
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        # Buffers (trained with running(momentum update))
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

    def __call__(self, x):
        # Calculate the forward pass
        if self.training:
            xmean = x.mean(0 , keepdim = True)
            xvar = x.var(0 , keepdim = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)# Normalizing to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters (self):
        return [self.gamma , self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters (self):
        return []

n_emb = 10 #the dimensionality of the characters embedding vector
n_hidden = 300 # number of neurons in the hidden layer of the MLP
g = torch.Generator(device=device).manual_seed(2147483647) # for reprodutivity

if device.type == 'cuda':
    C = torch.randn((vocab_size, n_emb), device=device)
else:
    C = torch.randn((vocab_size , n_emb), generator=g, device=device)
layers = [
   Linear(n_emb * block_size, n_hidden, bias=False, gain=5/3, device=device),BenchNorm(n_hidden, device=device),Tanh(),
   Linear(          n_hidden, n_hidden, bias=False, gain=5/3, device=device),BenchNorm(n_hidden, device=device),Tanh(),
   Linear(          n_hidden, n_hidden, bias=False, gain=5/3, device=device),BenchNorm(n_hidden, device=device),Tanh(),
   Linear(          n_hidden, n_hidden, bias=False, gain=5/3, device=device),BenchNorm(n_hidden, device=device),Tanh(),
   Linear(          n_hidden, n_hidden, bias=False, gain=5/3, device=device),BenchNorm(n_hidden, device=device),Tanh(),
   Linear(          n_hidden, vocab_size, bias=False, gain= 1.0, device=device),BenchNorm(vocab_size, device=device),
]
with torch.no_grad():
    # making the last layer less confident
    layers[-1].gamma *= 0.1
    #applying gain to all other layers # we commented this section because we use Xavier (Glorot) at the initialization and managing the gains inside of the layers them selves this way we reduce the stature at the activation

    # for layer in layers[: -1]:
    #     if isinstance(layer, Linear):
    #         layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

#same optimization with a little structure changes to match this PyTorch version and be compatible
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for step in range(max_steps):
    ix = torch.randint(0 ,Xtr.shape[0], (batch_size,) , generator = g, device=device)
    Xb , Yb = Xtr[ix], Ytr[ix]

    #forward pass
    emb = C[Xb]# embedding the characters into vectors
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
      x = layer(x)
    loss = F.cross_entropy(x, Yb)

    #backward pass
    for layer in layers:
        layer.out.retain_grad() #after debug would take out retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.01 if step < 25000 else 0.0001 #step's learning rate decay
    for p in parameters:
        p.data += -lr * (p.grad + 0.000001 * p.data)


    if step % 1000 == 0:
        print(f'{step:7d}/{max_steps:7d}, loss: {loss.item():.8f}')
    with torch.no_grad():
     ud.append([((lr * (p.grad + 0.000001 * p.data)).std() / p.data.std()).log10().item() for p in parameters])

    if step >= 50000:
     break

plt.figure(figsize=(20,10))
plt.imshow(h.detach().abs().cpu() > 0.99, cmap = 'gray', interpolation = 'nearest')
plt.show()

# for p in parameters:
#  print(f'{p.data.std().item()} test::')
# Visualizing the histogram
plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[: -1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hy, hx = torch.histogram(t.cpu() , density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
plt.show()


# Visualizing the histogram of grad
plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[: -1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hy, hx = torch.histogram(t.cpu() , density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
plt.show()


# Visualizing the update of grads by data ratio !!!!!!!!!!!!!!!!
plt.figure(figsize=(20,4))
legends = []
for i,p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0 , len(ud)], [-3 , -3], 'k')# these ratios should be ~1e-3, indicate on plt
plt.legend(legends)
plt.title('Update ratio')
plt.show()

# -- sampling from the model
g = torch.Generator(device=device).manual_seed(2147483647 + 10)

# Set all BatchNorm layers to evaluation mode
for layer in layers:
    if isinstance(layer, BenchNorm):
        layer.training = False

for _ in range(10):
    out = []
    context = [0] * block_size  # initialize with special '.' tokens

    while True:
        # Prepare input - get embeddings for current context
        emb = C[torch.tensor([context])]  # shape: [1, block_size, n_emb]

        # Forward pass through the network
        x = emb.view(emb.shape[0], -1)  # flatten to [1, block_size * n_emb]

        # Pass through each layer (same as training)
        for layer in layers:
            x = layer(x)

        # x now contains logits [1, vocab_size]
        probs = F.softmax(x, dim=1)

        # Sample from the distribution
        ix = torch.multinomial(probs, 1, generator=g).item()

        # Shift context window and append new character
        context = context[1:] + [ix]
        out.append(ix)

        # If we sample special '.' token, break
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))  # decode and print the generated word

# Set BatchNorm layers back to training mode (if you plan to train more)
for layer in layers:
    if isinstance(layer, BenchNorm):
        layer.training = True