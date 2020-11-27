import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, norder, ninp, nhid, nlayers,nonlin, dropout=0.2, tie_weights=False):
        super(FNNModel, self).__init__()
        self.ntoken = ntoken
        #self.norder = norder
        self.drop = nn.Dropout(dropout)
        self.model_type = 'FeedForward'
        self.window_size = ninp * (norder - 1)
        self.encoder = nn.Embedding(ntoken, ninp)

        if (nonlin=='relu'):
            self.nonlin = nn.ReLU()
        elif (nonlin=='tanh'):
            self.nonlin = nn.Tanh()
        elif (nonlin=='sigmoid'):
            self.nonlin = nn.Sigmoid()
        
        
        self.fnn = nn.Linear(self.window_size,nhid)
        

        self.decoder = nn.Linear(nhid, ntoken)


        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        #self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        emb = self.encoder(input).view(-1,self.window_size)
        #output= self.drop(self.fnn(emb))
        #output = self.nonlin(output)
        output = self.fnn(emb)
        output = self.drop(self.nonlin(output))
        decoded = self.decoder(output)
        #decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1)

