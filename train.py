import sys
import pdb
from model import seq2seq
#from evaluate import evaluate_model, model_perplexity
from data_loader import get_loader
import math
import numpy as np
import logging
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.autograd import Variable

dictionary = json.load(open('./data/dictionary.json'))
vocab_size = len(dictionary)

batch_size = 200

weight_mask = torch.ones(vocab_size).cuda()

criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

model = seq2seq(
    embedding_dim=100,
    vocab_size=vocab_size,
    hidden_size=100,
    batch_size=200,
    pad_token=dictionary['<PAD>'],
).cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

train_loader = get_loader('./data/train.src', './data/train.tgt', dictionary, batch_size)

for it in range(20):
    print(it)
    for i, batch in enumerate(train_loader):
        src, tgt = batch
        d_logit = model(src, tgt)
        optimizer.zero_grad()
        loss = criterion(
            d_logit.contiguous().view(-1, vocab_size),
            tgt.view(-1),
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

