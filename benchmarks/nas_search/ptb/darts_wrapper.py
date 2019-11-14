import sys
from benchmarks.model_def import BenchmarkDef
from multiprocessing import Queue, Process
import pickle
from benchmarks.nas_search.ptb.darts_trainer import DartsTrainer

# Set path for repo https://github.com/liamcli/darts
sys.path.append('/home/liamli4465/darts/rnn')
import genotypes
from genotypes import Genotype

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
import logging

class DartsSearch:
    def __init__(self, seed):
        np.random.seed(seed)
    def get_param_range(self, n, stochastic=True):
        configs = []
        for _ in range(n):
            n_nodes = genotypes.STEPS
            n_ops = len(genotypes.PRIMITIVES)
            arch = []
            for i in range(n_nodes):
                op = np.random.choice(range(1,n_ops))
                node_in = np.random.choice(range(i+1))
                arch.append((genotypes.PRIMITIVES[op], node_in))
            #concat = [i for i in range(genotypes.STEPS) if i not in [j[1] for j in arch]]
            concat = range(1,9)
            genotype = genotypes.Genotype(recurrent=arch, concat=concat)
            configs.append(genotype)
        return configs

class DartsWrapper(BenchmarkDef):
    def __init__(self, data_name, seed):
        self.data_name = data_name
        assert data_name in self.get_allowed_datasets()

        self.seed = seed
        self.search_space = self.model_params()

    def is_pbt_compatible(self):
        return False
    def is_cumulative_iters(self):
        return False
    def requires_R(self):
        True
    def requires_device(self):
        return False
    def set_device(self,device):
        self.device = device
    def set_R(self,max_units):
        self.max_units = max_units
    def get_R(self):
        return self.max_units
    def get_search_space(self):
        return self.search_space
    def get_allowed_datasets(self):
        return ['ptb']
    def model_params(self):
        params = {}
        params['genotype'] = DartsSearch(self.seed)
        return params

    def create_arm(self, arm_dir, params, combined_train=False, default=False):
        os.chdir(arm_dir)
        arm = {}
        if default:
            dirname = "default_arm"
        else:
            subdirs = next(os.walk('.'))[1]
            arm_num = len(subdirs)
            dirname = "arm" + str(arm_num)
            arm['seed'] = arm_num
            for hp in self.search_space.keys():
                val = params[hp]
                arm[hp] = val
        arm['name'] = dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        arm['dir'] = arm_dir + "/" + dirname
        if default:
            arm['genotype'] = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))
            arm['seed'] = 0
        else:
            for hp in self.search_space.keys():
                arm[hp] = params[hp]

        arm['results'] = []
        arm['epochs'] = 0
        return arm

    def solver_func(self,q,arm,n_units):
        trainer = DartsTrainer(arm)
        trainer.epoch = arm['epochs']
        overhead, val_perplexity, test_perplexity = trainer.train_epochs(n_units)
        q.put([overhead, -val_perplexity, -test_perplexity])

    def run_solver(self, arm, n_units):
        n_units = max(n_units - arm['epochs'], 0)
        logging.info('Training arm %d for %d epochs' % (arm['seed'], n_units))
        q = Queue()
        p = Process(target=self.solver_func, args=(q, arm, n_units))
        p.start()
        results = q.get()
        p.join()
        arm['epochs'] += n_units
        return results[0], results[1], results[2]

    def get_arm_iter(self, arm):
        return arm['epochs']

    def set_arm_iter(self, arm, epochs):
        arm['epochs'] = epochs

    def get_checkpoint_prefix(self):
        return 'checkpoint.incumbent'

def main():
    # Use for testing
    OUTPUTROOT='/home/liamli4465/results/asha_ptb'
    output_dir=os.path.join(OUTPUTROOT,'test')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model= DartsWrapper('ptb',261)
    params = {'genotype':model.search_space['genotype'].get_param_range(1)[0]}
    arm = model.create_arm(output_dir, params=params, default=False)
    # If resuming manually need to change these.
    #arm['dir'] = os.path.join(OUTPUTROOT,'eval/bracket401_arm13/arm0')
    #arm['epochs'] = 512
    model.set_R(300)
    overhead, val_acc, test_acc = model.run_solver(arm,1)
    overhead, val_acc, test_acc = model.run_solver(arm,2)

if __name__=="__main__":
    main()

