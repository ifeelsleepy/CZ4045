###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import math

import torch

import data
#import main

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)


is_feedforward_model = hasattr(model, 'model_type') and model.model_type == 'FeedForward'
        
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

model.eval()

with open(args.outf, 'w', encoding='utf-8') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            seq_len = 8 - 1
            start = i - seq_len

            if start < 0:
                dummy_token = corpus.dictionary.word2idx["<eos>"]
                dummy_tokens = torch.full((-start, input.size(1)), dummy_token).to(device)
                data = torch.cat((dummy_tokens, input[0:i]))
            else:
                data = input[i - seq_len:i]

            output = model(data)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

            # generated = corpus.tokenize('generated.txt')
            # generate_data = main.batchify(generated, main.eval_batch_size)

            # #Run on test data
            # gen_loss = main.evaluate(generate_data)
            # print( '=' * 89)
            # print('| End of training | generate  text loss {:5.2f} | generate text ppl {:8.2f}'.format(
            #     gen_loss, math.exp(gen_loss)))
            # print('=' * 89)