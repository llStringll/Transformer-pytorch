"""Vanilla transformer on WikiText2 LM task (GPU)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torchtext import data, datasets
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()

# this section for dataset and shit
print ("Preparing data...")
import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
TEXT = data.Field(tokenize=tokenize_en,init_token=BOS_WORD,eos_token=EOS_WORD,pad_token=BLANK_WORD)

MAX_LEN=200
Train, Val, Test = datasets.WikiText2.splits(TEXT)

MIN_FREQ=2
TEXT.build_vocab(Train,min_freq=MIN_FREQ)
BATCH_SIZE=5000

# train[0].text=train[0].text[0:200000]
# val[0].text=val[0].text[0:100000]
# test[0].text=test[0].text[0:15000]
print("Total corpus size:",len(train[0].text),len(val[0].text),len(test[0].text))

# BPTTIterator base class as in pytorch doc https://pytorch.org/text/_modules/torchtext/data/iterator.html
class BPTTIterator(data.BPTTIterator):
    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, bptt_len, **kwargs)

    def __iter__(self):
        text = self.dataset[0].text
        TEXTc = self.dataset.fields['text']
        TEXTc.eos_token = None
        text = text + ([TEXTc.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        datac = TEXTc.numericalize(
            [text], device=self.device)
        datac = datac.view(self.batch_size, -1).t().contiguous()
        datasetc = data.Dataset(examples=self.dataset.examples, fields=[('text', TEXTc), ('target', TEXTc)])
        while True:
            for i in range(0, len(self) * self.bptt_len - self.bptt_len, 1):
                # print (len(self))
                self.iterations += 1
                seq_len = min(2*self.bptt_len, len(datac) - i)
                batch_text = datac[i : i + int(seq_len/2)]
                batch_target = datac[i + int(seq_len/2) : i + seq_len]
                # print (batch_text.size(),batch_target.size())
                if TEXTc.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield data.Batch.fromvars(
                    datasetc, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return

train, val, test = BPTTIterator.splits(
    (Train, Val, Test),
    batch_size=BATCH_SIZE,
    bptt_len=MAX_LEN,
    repeat=False,
    device=torch.device(0),
    shuffle=True)

print ("Sample validation set:")
for i, batch in enumerate(val):
  # print (batch.text.size())
  for i in range(1, batch.text.size(0)):
      sym = TEXT.vocab.itos[batch.text.data[i, 1]]
      print(sym, end =" ")
  print("\nabove is sample text, below is sample target")
  print (batch.target.size())
  for i in range(1, batch.target.size(0)):
      sym = TEXT.vocab.itos[batch.target.data[i, 1]]
      print(sym, end =" ")
  print()
  print (batch.text.size())
  for i in range(1, batch.text.size(0)):
      sym = TEXT.vocab.itos[batch.text.data[i, 2]]
      print(sym, end =" ")
  print("\nabove is sample text, below is sample target")
  print (batch.target.size())
  for i in range(1, batch.target.size(0)):
      sym = TEXT.vocab.itos[batch.target.data[i, 2]]
      print(sym, end =" ")
  print()
  break

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.text))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.target) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class BatchIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    src, trg = batch.text.transpose(0, 1), batch.target.transpose(0, 1)
    return Batch(src, trg, pad_idx)

class Batch:
    def __init__(self,src,trg=None,pad=0):
        self.text=src
        self.src_mask=(src!=pad).unsqueeze(-2)
        # self.src_mask = self.make_src_mask(src,pad)
        if trg is not None:
            self.target = trg[:,:-1]
            self.trg_y = trg[:,1:]
            self.trg_mask = self.make_std_mask(self.target,pad)
            self.ntokens=(self.trg_y!=pad).data.sum()
    @staticmethod
    def make_std_mask(tgt,pad):
        tgt_mask=(tgt!=pad).unsqueeze(-2)
        tgt_mask=tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    @staticmethod
    def make_src_mask(src,pad):
        src_mask=(src!=pad).unsqueeze(-2)
        src_mask=src_mask & Variable(subsequent_mask(src.size(-1)).type_as(src_mask.data))
        return src_mask

def subsequent_mask(size):
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0

# model arch here on
print("Creating model arch...")
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,enc_out,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt), enc_out, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,vocab)
    def forward(self,x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-08):
        super(LayerNorm,self).__init__()
        self.sigma = nn.Parameter(torch.ones(features))
        self.mu = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.sigma*(x-mean)/(std+self.eps) + self.mu

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_fwd,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_fwd = feed_fwd
        self.sublayer = clones(SublayerConnection(size,dropout), 2)
        self.size = size
    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_fwd)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clones(layer, N)
        self.norm=LayerNorm(layer.size)
    def forward(self,x,enc_out,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,enc_out,src_mask,tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_fwd,dropout):
        super(DecoderLayer, self).__init__()
        self.size=size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_fwd = feed_fwd
        self.sublayer = clones(SublayerConnection(size,dropout),3)
    def forward(self,x,enc_out,src_mask,tgt_mask):
        m=enc_out
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x, self.feed_fwd)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,dff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.l1=nn.Linear(d_model,dff)
        self.l2=nn.Linear(dff,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.l2(self.dropout(F.relu(self.l1(x))))

# how nn.Embedding works - https://stackoverflow.com/questions/50747947/embedding-in-pytorch
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)

class PosEnc(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PosEnc,self).__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab,tgt_vocab,N=6,d_model=512,dff=2048,h=8,dropout=0.1):
    c=copy.deepcopy
    attn=MultiHeadedAttention(h,d_model)
    ff=PositionwiseFeedForward(d_model,dff,dropout)
    position=PosEnc(d_model,dropout)
    model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
    Decoder(DecoderLayer(d_model, c(attn), c(attn),
                         c(ff), dropout), N),
    nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
    Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def run_epoch(data_iter,model,loss_compute):
    start=time.time()
    total_tokens=0
    total_loss=0
    tokens=0
    for i,batch in enumerate(data_iter):
        # for i in range(0, batch.target.size(1)):
        #     sym = TEXT.vocab.itos[batch.target.data[0, i]]
        #     print(sym, end =" ")
        # print ()
        # for i in range(0, batch.text.size(1)):
        #     sym = TEXT.vocab.itos[batch.text.data[0, i]]
        #     print(sym, end =" ")
        # print()
        # print ("batch.text",batch.text.shape,"batch.target",batch.target.shape,"batch.textmask",batch.src_mask.shape,"batch.targetmask",batch.trg_mask.shape)
        # exit()
        out = model.forward(batch.text, batch.target, batch.src_mask, batch.trg_mask)
        loss=loss_compute(out,batch.trg_y,batch.ntokens)
        total_loss+=loss
        total_tokens+=batch.ntokens
        tokens+=batch.ntokens
        if (i+1)%100==0:
            elapsed=time.time()-start
            print("Batch No: %d Loss: %f Tokens per Sec: %f" %(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss/total_tokens

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx,as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    def __init__(self,generator,criterion,opt=None):
        self.generator=generator
        self.criterion=criterion
        self.opt=opt
    def __call__(self,x,y,norm):
        x=self.generator(x)
        loss=self.criterion(x.contiguous().view(-1,x.size(-1)),y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data*norm

class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data,
                                    requires_grad=self.opt is not None)]
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i+chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum() / normalize
            total += l.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

#applying greedy decoding for inference of the trained model
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    enc_out = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(enc_out, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

print ("Creating model...")
devices=[0]
pad_idx = TEXT.vocab.stoi["<blank>"]
model=make_model(len(TEXT.vocab),len(TEXT.vocab),N=6)
model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
model.generator.proj.weight = model.tgt_embed[0].lut.weight
model.cuda()
criterion=LabelSmoothing(size=len(TEXT.vocab),padding_idx=pad_idx,smoothing=0.1)
criterion.cuda()
train_itr=train
valid_itr=val
model_par = nn.DataParallel(model, device_ids=devices)

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

print ("Starting training...")
EPOCH=50
print ("Batch size:",BATCH_SIZE,"Total epochs:",EPOCH)
for epoch in range(EPOCH):
    model_par.train()
    print("Epoch:",epoch)
    run_epoch((rebatch(pad_idx,b) for b in train_itr),model_par,MultiGPULossCompute(model.generator, criterion,devices=devices, opt=model_opt))
    model_par.eval()
    print("Evaluating for this epoch on validation set")
    loss = run_epoch((rebatch(pad_idx,b) for b in valid_itr),model_par,MultiGPULossCompute(model.generator, criterion,devices=devices, opt=None))
    print ("Validation set loss : ",loss)
    # test run on valid set
    for i, batch in enumerate(valid_itr):
        src = batch.text.transpose(0, 1)[:1]
        src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TEXT.vocab.stoi["<s>"])
        print("Generated text 1:", end="\t")
        for i in range(1, out.size(1)):
            sym = TEXT.vocab.itos[out[0, i]]
            print(sym, end =" ")
        print()
        print("Source text 1:", end="\t")
        for i in range(1, batch.text.size(0)):
            sym = TEXT.vocab.itos[batch.text.data[i, 0]]
            print(sym, end =" ")
        print()
        print("Target text 1:", end="\t")
        for i in range(1, batch.target.size(0)):
            sym = TEXT.vocab.itos[batch.target.data[i, 0]]
            print(sym, end =" ")
        print()

        src = batch.text.transpose(0, 1)[1:2]
        src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TEXT.vocab.stoi["<s>"])
        print("Generated text 2:", end="\t")
        for i in range(1, out.size(1)):
            sym = TEXT.vocab.itos[out[0, i]]
            print(sym, end =" ")
        print()
        print("Source text 2:", end="\t")
        for i in range(1, batch.text.size(0)):
            sym = TEXT.vocab.itos[batch.text.data[i, 1]]
            print(sym, end =" ")
        print()
        print("Target text 2:", end="\t")
        for i in range(1, batch.target.size(0)):
            sym = TEXT.vocab.itos[batch.target.data[i, 1]]
            print(sym, end =" ")
        print()

        src = batch.text.transpose(0, 1)[2:3]
        src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TEXT.vocab.stoi["<s>"])
        print("Generated text 3:", end="\t")
        for i in range(1, out.size(1)):
            sym = TEXT.vocab.itos[out[0, i]]
            print(sym, end =" ")
        print()
        print("Source text 3:", end="\t")
        for i in range(1, batch.text.size(0)):
            sym = TEXT.vocab.itos[batch.text.data[i, 2]]
            print(sym, end =" ")
        print()
        print("Target text 3:", end="\t")
        for i in range(1, batch.target.size(0)):
            sym = TEXT.vocab.itos[batch.target.data[i, 2]]
            print(sym, end =" ")
        print()

        inpstr = "There was fire".split()
        src = torch.LongTensor([[TEXT.vocab.stoi[w] for w in inpstr]]).cuda()
        src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2).cuda()
        out = greedy_decode(model, src, src_mask,
                            max_len=200, start_symbol=TEXT.vocab.stoi["<s>"])
        print("Coustom Input:",inpstr)
        print("Generated text for custom input:", end="\t")
        for i in range(1, out.size(1)):
            sym = TEXT.vocab.itos[out[0, i]]
            print(sym, end =" ")
        print()
        break

# save chkpnt for inference
checkpoint = {'lenText': len(TEXT.vocab),
              'N': 6,
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model=make_model(checkpoint['lenText'],checkpoint['lenText'],N=checkpoint['N'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

# user sample input to a loaded model
MMMMModel=make_model(len(TEXT.vocab),len(TEXT.vocab))
MMMMModel = load_checkpoint('checkpoint.pth')
inpstr = "There was fire".split()
src = torch.LongTensor([[TEXT.vocab.stoi[w] for w in inpstr]])
src_mask = (src != TEXT.vocab.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(MMMMModel, src, src_mask,
                    max_len=500, start_symbol=TEXT.vocab.stoi["<s>"])

print("Input:",inpstr)
print("Generated text:", end="\t")
for i in range(1, out.size(1)):
    sym = TEXT.vocab.itos[out[0, i]]
    print(sym, end =" ")
print()
