
import tiktoken
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('gpu enabled') if torch.cuda.is_available() else print('training on cpu')
eval_iters = 200
torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
n_embd= 32

#### check unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# 第一步，enumerate 文本所有字符的集合, chars, 找到每个字母和其对应的index
stoi = {ch:i for i, ch in enumerate(chars)}
# 同时准备decode的字典，即所有index和其对应的文本
itos = {i:ch for i, ch in enumerate(chars)}

## encoder将字符转换成索引(int), 而decoder将索引转换回字符，所以需要两个key,value相反的字典

## encoder函数，输出输入文本中，每个字符对应的索引
encode = lambda s: [stoi[c] for c in s]
## decoder函数，输出输入字符中，每个整数对应的字符
decode = lambda l: ''.join([itos[i] for i in l])



data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:10])

#### 构建训练和验证数据集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



#### Data Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    ### 使用cuda时，确保数据被转移到cuda的设备上
    x, y = x.to(device), y.to(device)
    return x, y


##############################
######## 特制损失函数 #########
##############################

### 使用no_grad()函数，当目标函数不会进行反向传播运算时，这样目标函数将不会自动储存运算结果
@torch.no_grad()
def estimate_loss():
    out = {}
    ## 开启模型的评估模式: 此模式下，dropout层会停止dropout，所有神经元参与运算，BN层使用运行统计量而非当前batch统计量
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    ## 评估完成后要切换回训练模式, eval(), train()都是nn.Module类自带函数，且往往成对使用
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


### 这里客制化一个最基础的语言模型，继承于nn.Module类
class BigramLanguageModel(nn.Module):
    ## 改写init函数，初始化一个嵌入层
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)


    ## 改写forward函数，向前传播方法改为查询输入值在嵌入层中的对应向量
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) ## B,T,n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            ## 修剪idx，输入矩阵的形状，使得它不超过block size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


############ 设置优化器 #############
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#### 优化损失

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step{iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}')

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    ## 每个新批次开始前，重置梯度为0，避免梯度累积
    optimizer.zero_grad(set_to_none=True)
    ## 对损失使用反向传播计算
    loss.backward()
    ## 根据优化器所使用的算法来更新模型参数
    optimizer.step()
    
### 生成预测文本时，也需要再device上生成
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))



