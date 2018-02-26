import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
import numpy as np


class seq2seq(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size, batch_size, pad_token):
        super(seq2seq, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.pad_token = pad_token

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token,
        )

        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            1,
            batch_first = True,
            dropout = 0
        )

        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            1,
            batch_first = True,
            dropout = 0
        )

        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

        self.decoder2vocab = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        batch = src.size(0)
        self.h = Variable(torch.zeros(1, batch_size, self.hidden_size))
        self.c = Variable(torch.zeros(1, batch_size, self.hidden_size))

        src_h, (src_h_t, src_c_t) = self.encoder(src_emb, self.h, self.c)
        h_t = src_h_t[-1]
        c_t = src_c_t[-1]

        d_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        tgt_h, (_, _) = self.decoder(
            tgt_emb,
            (
                d_init_state.view(1, d_init_state.size(0), d_init_state.size(1)),
                c_t.view(1, c_t.size(0), c_t.size(1))
            )
        )

        tgt_h_reshape = tgt_h.contiguous().view(
            tgt_h.size(0) * tgt_h.size(1),
            tgt_h.size(2)
        )

        d_logit = self.decoder2vocab(tgt_h_reshape)
        d_logit = d_logit.view(
            tgt_h.size(0),
            tgt_h.size(1),
            d_logit.size(1)
        )

        return d_logit

    def decode(self, logits):
        l_reshape = logits.view(-1, self.vocab_size)
        prob = func.softmax(l_reshape)
        prob = prob.view(logits.size()[0], logits.size()[1], logits.size()[2])
        return prob

