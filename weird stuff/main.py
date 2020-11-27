import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='FeedForward',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--save_data', type=str, default='train_data.txt',
                    help='path to save the training results')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--norder', type=int, default=8,
                    help='context size in feed-forward model; the number of heads in the transformer model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--SGD', type=bool, default=False,
                    help='run on SGD optimization')
parser.add_argument('--train_log_interval', type=int, default=1000,
                    help='train log interval')
parser.add_argument('--nonlin', type=str, default='tanh',
                    help='Nonlinearity function')
parser.add_argument('--verbose', type=bool, default=True,
                    help='Verbose ')


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if (args.model == 'Transformer'):
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
elif (args.model == 'FeedForward'):
    model_1 = model.FNNModel(ntokens,args.norder, args.emsize, args.nhid, args.nlayers, args.nonlin, args.dropout, args.tied).to(device)
    print(model_1)
elif (args.model =='FeedForward_1'):
    model_1 = model.FNNModel(ntokens, args.norder, 90, 90, args.nlayers, 0.3, args.tied).to(device)
    print(model_1)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

#Instantiate loss class
criterion = nn.NLLLoss()

# Instantiate opimizer class
if (args.SGD == True):
    print("Optimizing based on SGD")
    learning_rate = args.lr
    optimizer = torch.optim.SGD(model_1.parameters(), lr=learning_rate)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = (min(args.bptt, len(source) - 1 - i))
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_ngrams(source,i):
    seq_len = args.norder - 1
    start = i - seq_len
    if start <0:
        dummy_token = corpus.dictionary.word2idx["<eos>"]
        dummy_tokens = torch.full((-start,source.size(1)), dummy_token).to(device)
        data = torch.cat((dummy_tokens,source[0:i]))
    else:
        data = source[i-seq_len:i]

    target = source[i+1].view(-1)
    return data, target

    # data = source[i:i+seq_len]
    # target = source[i+1: i+1+args.norder].view(-1)
    # return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model_1.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    total_size = 0

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1):
            data, targets = get_ngrams(data_source, i)
            output = model_1(data)
            #output = output.view(-1, ntokens)
            total_size+=len(data)
            total_loss += len(data) * criterion(output, targets).item()

            # else:
            #     output = model(data)
            #     hidden = repackage_hidden(hidden)

    return total_loss / total_size


def train(shorter_data=False):
    # Turn on training mode which enables dropout.
    model_1.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    source = train_data[:1000] if shorter_data else train_data

    for batch, i in enumerate(range(0, source.size(0) - 1)):
        data, targets = get_ngrams(source, i)
        if (args.SGD == True):
                # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
        else:
            model_1.zero_grad()

        output = model_1(data)
        loss = criterion(output, targets)
        #print(loss)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), args.clip)
        for p in model_1.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.train_log_interval == 0 and batch > 0 and args.verbose:
            #print(total_loss)
            cur_loss = total_loss / args.train_log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(source), lr,
                elapsed * 1000 / args.train_log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

if args.tied:
    emsizes = [10,30,90,270]
    nhids = [None]
else:
    emsizes = [10,30,90,270]
    nhids = [10,30,90,270]

dropouts = [0, 0.2, 0.5]
nonlins = ['tanh','relu','sigmoid']
best_val_loss = 999
epochs = 3
lr = args.lr

# print('-' * 89)
# print("Hyperparameters tuning using val loss after {:1d} epochs".format(epochs))
# print('-' * 89)

# args.verbose = False

# try:
#     for emsize in emsizes:
#         for nhid in nhids:
#             for dropout in dropouts:
#                 for nonlin in nonlins:
#                     if nhid is None:
#                         nhid = emsize

                    
#                     torch.manual_seed(args.seed)
#                     args.model = 'FeedForward'
#                     args.nonlin = nonlin
#                     args.nhid = nhid
#                     args.emsize =emsize
#                     args.dropout = dropout
#                     model_1 = model.FNNModel(ntokens,args.norder, args.emsize, args.nhid, args.nlayers, args.nonlin, args.dropout, args.tied).to(device)
                    
#                     for epoch in range(1, epochs+1):
                        
#                         train(shorter_data=True)
#                         val_loss = evaluate(val_data[:100])
                        
#                         if (val_loss < best_val_loss):
#                             best_val_loss = val_loss
#                             best_var = [emsize, nhid, nonlin, dropout]
#                         else:
#                             lr /= 4.0
                    
#                     print("| emsize {:3d} | nhid {:3d} | dropout {:02.1f} | nonlin: {} |val loss {:02.4f} ".format(emsize, nhid, dropout, nonlin, val_loss))

                    
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')

# def export_onnx(path, batch_size, seq_len):
#     print('The model is also exported in ONNX format at {}'.
#           format(os.path.realpath(args.onnx_export)))
#     model_1.eval()
#     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
#     hidden = model_1.init_hidden(batch_size)
#     torch.onnx.export(model_1, (dummy_input, hidden), path)

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


# # Loop over epochs.
# lr = args.lr
# val_losses = []
# train_losses = []
# torch.manual_seed(args.seed)

# args.verbose = True
# args.model = 'FeedForward_1'
# for i in range(0, len(best_var)):
#       print(best_var[i])
# model_1 = model.FNNModel(ntokens, args.norder,best_var[0],best_var[1], args.nlayers, best_var[2], best_var[3],args.tied).to(device)
# print(model_1)



# At any point you can hit Ctrl + C to break out of training early.
saved_data =[]
try:
    lr = args.lr
    best_val_loss = None
    val_losses = []
    train_losses = []

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        train_loss = evaluate(train_data)
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        saved_data.append('end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                             val_loss, math.exp(val_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model_1, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save_data, 'w', encoding='utf-8') as outf:
    for i in range(len(saved_data)):
        outf.write(saved_data[i]+',')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model_1 = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

