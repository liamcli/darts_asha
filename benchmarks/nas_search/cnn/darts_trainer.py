import sys
sys.path.append('/home/liamli4465/darts/cnn')
from model import NetworkCIFAR as Network
import utils

import time
import math
import copy
import logging
import random
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsTrainer:
    def __init__(self, arm):
        args = {}
        args['data'] = '/home/liamli4465/darts/data/'
        args['batch_size'] = 96
        args['learning_rate'] = 0.025
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['report_freq'] = 50
        args['gpu'] = 0
        args['init_channels'] = 36
        args['layers'] = 20
        args['auxiliary'] = True
        args['auxiliary_weight'] = 0.4
        args['cutout'] = True
        args['cutout_length'] = 16
        args['drop_path_prob'] = 0.2
        args['grad_clip'] = 5
        args['epochs'] = 300
        args['save'] = arm['dir']
        args['seed'] = arm['seed']
        args['arch'] = arm['genotype']
        args['cuda'] = True
        args = AttrDict(args)
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)


        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=False, transform=valid_transform)

        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

        self.epoch = 0

    def model_save(self, fn, to_save):
        if self.epoch % 128 == 0:
            with open(os.path.join(self.args.save, "checkpoint-incumbent-%d" % self.epoch), 'wb') as f:
                torch.save(to_save, f)

        with open(fn, 'wb') as f:
            torch.save(to_save, f)

    def model_load(self, fn):
        with open(fn, 'rb') as f:
            self.model, self.optimizer, self.scheduler, rng_state, cuda_state = torch.load(f)
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_state)

    def model_resume(self, filename):
            logging.info('Resuming model from %s'%filename)
            self.model_load(filename)
            self.optimizer.param_groups[0]['lr'] = self.args.learning_rate

    def train_epochs(self, epochs):
        args = self.args

        resume_filename = os.path.join(self.args.save, "checkpoint.incumbent")
        if os.path.exists(resume_filename):
            self.model_resume(resume_filename)
            logging.info('Loaded model from checkpoint')
        else:
            model = Network(args.init_channels, 10, args.layers, args.auxiliary, args.arch)
            model = model.cuda()
            self.model = model

            logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay
                )
            self.optimizer = optimizer

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(args.epochs))
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

        best_val = 0

        for epoch in range(epochs):
          self.scheduler.step()
          logging.info('epoch %d lr %e', self.epoch, self.scheduler.get_lr()[0])
          self.model.drop_path_prob = args.drop_path_prob * self.epoch / args.epochs

          train_acc, train_obj = self.train_epoch()
          logging.info('train_acc %f', train_acc)

          valid_acc, valid_obj = self.infer()
          if valid_acc > best_val:
              best_val = valid_acc
              self.model_save(os.path.join(args.save, 'checkpoint.incumbent'), [self.model, self.optimizer, self.scheduler, torch.get_rng_state(), torch.cuda.get_rng_state()])
              logging.info('Saving new best model!')

          logging.info('valid_acc %f', valid_acc)
        return 0, best_val,best_val

    def train_epoch(self):
      args = self.args
      objs = utils.AvgrageMeter()
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()
      self.model.train()

      for step, (input, target) in enumerate(self.train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        self.optimizer.zero_grad()
        logits, logits_aux = self.model(input)
        loss = self.criterion(logits, target)
        if args.auxiliary:
          loss_aux = self.criterion(logits_aux, target)
          loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
        self.optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return top1.avg, objs.avg


    def infer(self):
      objs = utils.AvgrageMeter()
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()
      self.model.eval()

      for step, (input, target) in enumerate(self.valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits, _ = self.model(input)
        loss = self.criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % self.args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return top1.avg, objs.avg

