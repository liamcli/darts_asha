import itertools
import pickle
import time
from functools import partial
import Trial
import numpy
import os
import sys
import logging


def _get_metadata(trial, key):
  return trial.metadata[key]

def _set_metadata(trial, key, value):
  trial.metadata[key] = value

class Conductor(object):
  """The conductor acts as an intermediary between the stateless policy and the workers
  """
  def __init__(self, policy):
    self.policy = policy
    self.completed_trials = {}
    self.pending_trials = {}
    self.trials_to_stop = {}
    self.trial_id_counter = itertools.count()
  def resume(self,filename):
    self.completed_trials = pickle.load(open(filename,'rb'))
    self.trial_id_counter = itertools.count(start=max(self.completed_trials.keys())+1)

  def get_suggestion(self):
    trial = self.policy.GetNewSuggestions(1, self.completed_trials.values(), self.pending_trials.values())[0]
    trial.id = next(self.trial_id_counter)
    self.pending_trials[trial.id] = trial
    return trial

  def report_done(self, trial, verbose=True):
    if 'report_done' in dir(self.policy):
        self.policy.report_done(trial)
    trial.status = 4
    del self.pending_trials[trial.id]
    self.completed_trials[trial.id] = trial
    if verbose == True:
        print(self.policy.summary(self.completed_trials.values(), self.pending_trials.values()))


class RandomSamplingPolicy(object):
  def __init__(self, arm_generator, termination_record):
    self.arm_generator = arm_generator
    self.termination_record = termination_record

  def GetNewSuggestions(self, num_suggestions_hint, completed_trials, pending_trials):
    trials = []
    for _ in range(num_suggestions_hint):
      trial = Trial.Trial()
      trial.metadata['termination_record'] = self.termination_record
      trial.parameters = self.arm_generator()
      trials.append(trial)
    return trials
class Measurement(object):
  def __init__(self, steps, objective_value,test_acc=0.0):
    self.objective_value = objective_value
    self.steps = steps
    self.test_acc = test_acc


class Model_Job(object):
  def __init__(self, module_name,data_name,output_dir,seed,max_records,device):
    #self.input_dir = input_dir
    self.output_dir = output_dir

    model_fn = self.model_mapping(module_name)

    model = model_fn(data_name, seed)

    model.set_R(max_records)
    if model.requires_device():
      model.set_device(device)

    self.model = model
    self.params = model.model_params()
    self.max_records = max_records

  def model_mapping(self, module_name):
    if module_name == 'darts_ptb':
      from benchmarks.nas_search.ptb.darts_wrapper import DartsWrapper
      return DartsWrapper
    elif module_name == 'darts_cnn':
      from benchmarks.nas_search.cnn.darts_wrapper import DartsWrapper
      return DartsWrapper
    else:
      raise NotImplementedError

  def train(self,trial,t_0,duration, interval=5000):
    print("Remaining time is %0.2f" % ((duration - (time.time()-t_0))/60.0))
    start_time = time.time()
    iters = 0
    # This flag indicates whether the resource will be interpreted as the desired total
    # amount trained.  For PBT, this should always be true.
    if self.model.is_cumulative_iters():
        if trial.measurements:
          iters = trial.measurements[-1].steps
    target_iters = trial.metadata['termination_record']
    trial.timing['comp_start']=time.time() - start_time
    train_iters = max(int((target_iters - iters)/interval) + 1, 1)
    tmp_iter = iters
    val_acc = 0
    test_acc = 0
    trial.timing['t_measurements'] = []
    for i in range(train_iters):
        tmp_iter = int(min(target_iters,tmp_iter + interval))
        if time.time() - t_0 < duration:
            train_loss,val_acc,test_acc = self.model.run_solver(trial.parameters, tmp_iter)
            print('training_loss: %.3f, val_acc: %.3f, test_acc: %.3f' % (train_loss, val_acc, test_acc))
            measurement = Measurement(tmp_iter, val_acc,test_acc)
            trial.measurements.append(measurement)
            trial.timing['t_measurements'].append(time.time() - start_time)
    trial.timing['comp_end']=time.time()-start_time
    return trial
  def arm_generator(self, params=None):
      if params is None:
          params = {}
          for hp in self.params.keys():
              params[hp] = self.params[hp].get_param_range(1,stochastic=True)[0]
      return self.model.create_arm(self.output_dir,params)

def setup_logger(output_dir):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
