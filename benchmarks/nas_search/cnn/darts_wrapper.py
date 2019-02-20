import sys
from benchmarks.model_def import BenchmarkDef
from multiprocessing import Queue, Process
import pickle
from benchmarks.nas_search.cnn.darts_trainer import DartsTrainer

# Set path for repo https://github.com/liamcli/darts
sys.path.append('/home/liamli4465/darts/cnn')
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
        self.arch_steps = 4
    def get_param_range(self, n, stochastic=True):
        configs = []
        op_dict = {
            0: 'none',
            1: 'max_pool_3x3',
            2: 'avg_pool_3x3',
            3: 'skip_connect',
            4: 'sep_conv_3x3',
            5: 'sep_conv_5x5',
            6: 'dil_conv_3x3',
            7: 'dil_conv_5x5'
            }
        for _ in range(n):
            k = sum(1 for i in range(self.arch_steps) for n in range(2+i))
            num_ops = len(genotypes.PRIMITIVES)
            n_nodes = self.arch_steps

            normal = []
            reduction = []
            for i in range(n_nodes):
                ops = np.random.choice(range(num_ops), 4)
                nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
                nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
                normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
                reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

            arch = (normal, reduction)
            darts_arch = [[], []]
            i=0
            for cell in arch:
                for n in cell:
                    darts_arch[i].append((op_dict[n[1]], n[0]))
                i += 1
            genotype = Genotype(normal=darts_arch[0], normal_concat=[2,3,4,5], reduce=darts_arch[1], reduce_concat=[2,3,4,5])
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
        return ['cifar10']
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
            arm['genotype'] = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
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
        overhead, val_acc, _ = trainer.train_epochs(n_units)
        q.put([overhead, val_acc, val_acc])

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
    output_dir=os.path.join(OUTPUTROOT,'asha_cnn/test_default')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model= DartsWrapper('ptb',261)
    params = {'genotype':model.search_space['genotype'].get_param_range(1)[0]}
    arm = model.create_arm(output_dir, params=params, default=False)
    # If resuming manually need to change these.
    arm['seed'] = 2
    arm['dir'] = os.path.join(OUTPUTROOT,'asha_cnn/test_default/arm2')
    #arm['epochs'] = 512
    model.set_R(300)
    overhead, val_acc, test_acc = model.run_solver(arm,1)
    overhead, val_acc, test_acc = model.run_solver(arm,2)

if __name__=="__main__":
    main()

