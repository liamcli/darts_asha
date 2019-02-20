import sys
sys.path.append('/home/liamli4465/darts/rnn')
from model import RNNModel
import genotypes
import data
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

import time
import math
import numpy as np
import torch
import copy
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import gc

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsTrainer():
    def __init__(self, arm):
        # Default params for eval network
        args = {'emsize':850, 'nhid':850, 'nhidlast':850, 'dropoute':0.1, 'wdecay':8e-7}

        args['data'] = '/home/liamli4465/darts/data/penn'
        args['lr'] = 20
        args['clip'] = 0.25
        args['batch_size'] = 64
        args['search_batch_size'] = 256*4
        args['small_batch_size'] = 64
        args['bptt'] = 35
        args['dropout'] = 0.75
        args['dropouth'] = 0.25
        args['dropoutx'] = 0.75
        args['dropouti'] = 0.2
        args['seed'] = arm['seed']
        args['nonmono'] = 5
        args['log_interval'] = 50
        args['save'] = arm['dir']
        args['alpha'] = 0
        args['beta'] = 1e-3
        args['max_seq_length_delta'] = 20
        args['unrolled'] = True
        args['gpu'] = 0
        args['cuda'] = True
        args['genotype'] = arm['genotype']
        args = AttrDict(args)
        self.args = args
        self.epoch = 0

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

        corpus = data.Corpus(args.data)
        self.corpus = corpus

        self.eval_batch_size = 10
        self.test_batch_size = 1

        self.train_data = batchify(corpus.train, args.batch_size, args)
        self.search_data = batchify(corpus.valid, args.search_batch_size, args)
        self.val_data = batchify(corpus.valid, self.eval_batch_size, args)
        self.test_data = batchify(corpus.test, self.test_batch_size, args)

        self.ntokens = len(corpus.dictionary)

    def model_save(self, fn, to_save):
        if self.epoch % 150 == 0:
            with open(os.path.join(self.args.save, "checkpoint-incumbent-%d" % self.epoch), 'wb') as f:
                torch.save(to_save, f)

        with open(fn, 'wb') as f:
            torch.save(to_save, f)

    def model_load(self, fn):
        with open(fn, 'rb') as f:
            self.model, self.optimizer, rng_state, cuda_state = torch.load(f)
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_state)

    def model_resume(self, filename):
            logging.info('Resuming model from %s'%filename)
            self.model_load(filename)
            self.optimizer.param_groups[0]['lr'] = self.args.lr
            for rnn in self.model.rnns:
                rnn.genotype = self.args.genotype

    def train_epochs(self, epochs):
        args = self.args
        resume_filename = os.path.join(self.args.save, "checkpoint.incumbent")
        if os.path.exists(resume_filename):
            self.model_resume(resume_filename)
            logging.info('Loaded model from checkpoint')
        else:
            self.model = RNNModel(self.ntokens, args.emsize, args.nhid, args.nhidlast,
                   args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute, genotype=args.genotype)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.wdecay)

        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))
        logging.info('initial genotype:')
        logging.info(self.model.rnns[0].genotype)

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

        self.model = self.model.cuda()
        # Loop over epochs.
        lr = args.lr
        best_val_loss = []
        stored_loss = 100000000

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(epochs):
                epoch_start_time = time.time()
                self.train()
                if 't0' in self.optimizer.param_groups[0]:
                    tmp = {}
                    for prm in self.model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = self.optimizer.state[prm]['ax'].clone()

                    val_loss2 = self.evaluate(self.val_data)
                    logging.info('-' * 89)
                    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            self.epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                    logging.info('-' * 89)

                    if val_loss2 < stored_loss:
                        self.model_save(os.path.join(args.save, 'checkpoint.incumbent'), [self.model, self.optimizer, torch.get_rng_state(), torch.cuda.get_rng_state()])
                        logging.info('Saving Averaged!')
                        stored_loss = val_loss2

                    for prm in self.model.parameters():
                        prm.data = tmp[prm].clone()

                else:
                    val_loss = self.evaluate(self.val_data, self.eval_batch_size)
                    logging.info('-' * 89)
                    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                      self.epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                    logging.info('-' * 89)

                    if val_loss < stored_loss:
                        self.model_save(os.path.join(args.save, 'checkpoint.incumbent'), [self.model, self.optimizer, torch.get_rng_state(), torch.cuda.get_rng_state()])
                        logging.info('Saving model (new best validation)')
                        stored_loss = val_loss

                    if (self.epoch > 75 and 't0' not in self.optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]))):
                        logging.info('Switching to ASGD')
                        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                    best_val_loss.append(val_loss)

        except Exception as e:
            logging.info('-' * 89)
            logging.info(e)
            logging.info('Exiting from training early')
            return 0, 10000, 10000

        # Load the best saved model.
        self.model_load(os.path.join(args.save, 'checkpoint.incumbent'))

        # Run on test data.
        val_loss = self.evaluate(self.val_data, self.eval_batch_size)
        logging.info(math.exp(val_loss))
        test_loss = self.evaluate(self.test_data, self.test_batch_size)
        logging.info('=' * 89)
        logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        logging.info('=' * 89)

        return 0, math.exp(val_loss), math.exp(test_loss)


    def train(self):
        args = self.args
        corpus = self.corpus
        total_loss = 0
        start_time = time.time()
        hidden = [self.model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
        batch, i = 0, 0

        while i < self.train_data.size(0) - 1 - 1:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            seq_len = min(seq_len, args.bptt + args.max_seq_length_delta)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            self.model.train()
            data, targets = get_batch(self.train_data, i, args, seq_len=seq_len)

            self.optimizer.zero_grad()

            start, end, s_id = 0, args.small_batch_size, 0
            while start < args.batch_size:
                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden[s_id] = repackage_hidden(hidden[s_id])

                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.model(cur_data, hidden[s_id], return_h=True)
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                loss = raw_loss
                # Activiation Regularization
                if args.alpha > 0:
                  loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss *= args.small_batch_size / args.batch_size
                total_loss += raw_loss.data * args.small_batch_size / args.batch_size
                loss.backward()

                s_id += 1
                start = end
                end = start + args.small_batch_size

                gc.collect()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()

            # total_loss += raw_loss.data
            self.optimizer.param_groups[0]['lr'] = lr2

            if np.isnan(total_loss[0]):
              raise

            #if batch % args.log_interval == 0 and batch > 0:
            #    cur_loss = total_loss[0] / args.log_interval
            #    elapsed = time.time() - start_time
            #    logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #            'loss {:5.2f} | ppl {:8.2f}'.format(
            #        self.epoch, batch, len(self.train_data) // args.bptt, self.optimizer.param_groups[0]['lr'],
            #        elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            #    total_loss = 0
            #    start_time = time.time()
            batch += 1
            i += seq_len
        self.epoch += 1

    def evaluate(self, data_source, batch_size=10):
	# Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.args.bptt):
            data, targets = get_batch(data_source, i, self.args, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = self.model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)
        return total_loss[0] / len(data_source)

