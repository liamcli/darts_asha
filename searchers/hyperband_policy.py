import numpy
from numpy import log, argmin
import Trial as Trial
from hp_utils import RandomSamplingPolicy, Conductor, Model_Job, _get_metadata, _set_metadata

class Bracket(object):
  def __init__(self, min_records, trials, num_halving_rounds=float('inf'), eta=0, bracket_id=None, goal_maximize=False,pending_trials=None, fix_quantiles=False):
    self._min_records = min_records
    self._max_records = min_records * eta ** num_halving_rounds
    self._num_halving_rounds = num_halving_rounds
    self._eta = eta
    self._bracket_id = bracket_id
    self._goal_maximize = goal_maximize
    # Total budget
    self._total_records = 0
    self._ladder = [[]]
    self._fix_quantiles = fix_quantiles
    self._pending_trials = {}
    for t in pending_trials:
      self._pending_trials[(_get_metadata(t,'hyperband_id'),_get_metadata(t,'termination_record'))]= True
    self._populate_bracket(trials)
  @property
  def total_records(self):
    return self._total_records
  @property
  def min_records(self):
    return self._min_records
  @property
  def ladder(self):
    return self._ladder
  @property
  def num_halving_rounds(self):
    return self._num_halving_rounds
  @property
  def num_in_lowest_rung(self):
    return len(self._ladder[0])
  @property
  def num_fully_trained(self):
    num_pending = len([key for key in self._pending_trials if key[1] == self._max_records])
    if len(self._ladder)<= self._num_halving_rounds:
      return num_pending
    return num_pending + len(self._ladder[self._num_halving_rounds])
  def _populate_bracket(self, trials):
    for trial in trials:
      if trial.measurements:
        records = trial.measurements[-1].steps
        self._total_records += records
        rung = int(round(log(records/self._min_records)/log(self._eta)))
        while len(self._ladder) <= rung:
          self._ladder.append([])
        self._ladder[rung].append(trial)
    for rung in range(len(self._ladder)):
      candidate_trials = [trial for trial in self._ladder[rung] if trial.status == Trial.COMPLETE and not trial.trial_infeasible]
      if self._goal_maximize:
        candidate_trials = sorted(candidate_trials, key=lambda x: -max(m.objective_value for m in x.measurements))
      else:
        candidate_trials = sorted(candidate_trials, key=lambda x: min(m.objective_value for m in x.measurements))
      self._ladder[rung] = candidate_trials
      if self._fix_quantiles and rung >= 1:
        trials_below = self._ladder[rung-1][0:int(len(self._ladder[rung-1])/self._eta)]
        ids_to_advance = [_get_metadata(trial,'hyperband_id') for trial in trials_below]
        for t in reversed(range(len(self._ladder[rung]))):
          trial = self._ladder[rung][t]
          if _get_metadata(trial,'hyperband_id') not in ids_to_advance:
            #print("Deleting configuration %d from rung %d" % (_get_metadata(trial,'hyperband_id'),rung))
            #print(len(self._ladder[rung]))
            del self._ladder[rung][t]
            #print(len(self._ladder[rung]))

  def get_trial(self):
    # identify the best rung to advance a trial from
    trial_to_advance = None
    #bracket_depth = min(len(self._ladder), self._num_halving_rounds)
    bracket_depth = len(self._ladder)
    trials_already_advanced = {}
    if len(self._ladder) > self._num_halving_rounds:
      trials_already_advanced.update({_get_metadata(trial, 'hyperband_id'): True for trial in self._ladder[-1]})
    for rung in reversed(range(bracket_depth)):
      candidate_trials = self._ladder[rung]
      if len(candidate_trials) >= self._eta:
        #each rung already sorted when populating brackets
        candidate_trials = candidate_trials[0:int(len(self._ladder[rung])/self._eta)]
        #print([_get_metadata(t,'hyperband_id') for t in candidate_trials])
        while candidate_trials:
          trial = candidate_trials.pop(0)
          hyperband_id = _get_metadata(trial, 'hyperband_id')
          if not trials_already_advanced.get(hyperband_id, False) and (hyperband_id,_get_metadata(trial,'termination_record')*self._eta) not in self._pending_trials:
            trial_to_advance = trial
            break
      if trial_to_advance is not None:
        break
      trials_already_advanced.update({_get_metadata(trial, 'hyperband_id'): True for trial in self._ladder[rung]})

    # advance an existing trial or initialize a new one
    if trial_to_advance:
      parameters = trial_to_advance.parameters
      measurements = trial_to_advance.measurements
      bracket_id = _get_metadata(trial_to_advance, 'bracket_id')
      hyperband_id = _get_metadata(trial_to_advance, 'hyperband_id')
      termination_record = self._eta*_get_metadata(trial_to_advance, 'termination_record')
    else:
      parameters = []
      measurements = []
      bracket_id = self._bracket_id
      hyperband_id = -1
      termination_record = self._min_records
    trial = Trial.Trial()
    trial.status = Trial.REQUESTED
    trial.parameters = parameters
    trial.measurements.extend(measurements)
    _set_metadata(trial, 'bracket_id', bracket_id)
    _set_metadata(trial, 'hyperband_id', hyperband_id)
    _set_metadata(trial, 'termination_record', termination_record)
    self._total_records += termination_record
    return trial

  def bracket_string(self):
    brk_str = ''
    def smean(S):
      if len(S)==0:
        return float('NaN')
      else:
        return numpy.mean(S)

    brk_str += 'r_i\tn_i\tbest objective_value\n---------------------\n'
    for idx,rung in enumerate(self._ladder):
      vals = []
      if rung and self._goal_maximize:
        vals = sorted([-max(x.objective_value for x in trial.measurements) for trial in rung if trial.measurements])
        vals = ['%.2f' % -x for x in vals]
      elif rung:
        vals = sorted([min(x.objective_value for x in trial.measurements) for trial in rung if trial.measurements])
        vals = ['%.2f' % x for x in vals]
      brk_str += '%.0f/%.0f, %d, %s\n' % (round(self._min_records*self._eta**idx),
                                     smean([trial.measurements[-1].steps for trial in rung if trial.measurements]),
                                     len(vals),
                                     str(vals))
    if len(self._ladder[-1]):
      brk_str += 'best trial is: %d\n' % (_get_metadata(self._ladder[-1][0],'hyperband_id'))
    return brk_str


class HyperbandPolicy(object):
  def __init__(self, delegate_policy, max_records, eta=3, max_halving_rounds=3, goal_maximize=False):
    self._delegate_policy = delegate_policy
    self._eta = eta
    self._min_records = max_records*self._eta**(-max_halving_rounds)
    self._max_halving_rounds = max_halving_rounds
    self._goal_maximize = goal_maximize
    self._max_records = max_records

  def _construct_brackets(self, completed_trials, pending_trials):
    all_trials = set(completed_trials).union(set(pending_trials))
    num_brackets = self._max_halving_rounds+1
    next_hyperband_id = -1
    stratified_trials = [[] for _ in range(num_brackets)]
    for trial in all_trials:
      stratified_trials[_get_metadata(trial, 'bracket_id')].append(trial)
      next_hyperband_id = max(next_hyperband_id, _get_metadata(trial, 'hyperband_id'))
    next_hyperband_id += 1
    brackets = []
    for s in range(num_brackets):
      bracket = Bracket(self._min_records*self._eta**(self._max_halving_rounds - s),
                        stratified_trials[s],
                        s,
                        self._eta,
                        s,
                        self._goal_maximize,
                        [trial for trial in pending_trials if _get_metadata(trial,'bracket_id')==s])
      brackets.append(bracket)
    return brackets, next_hyperband_id

  def GetNewSuggestions(self, num_suggestions_hint, completed_trials, pending_trials):
    # brackets reconstructed each time, probably can modify with each new result
    brackets, next_hyperband_id = self._construct_brackets(completed_trials, pending_trials)
    trials = []
    for _ in range(num_suggestions_hint):
      bracket_budget = [(idx, numpy.floor(bracket.num_fully_trained * (bracket.num_halving_rounds + 1) / (self._max_halving_rounds + 1))) for idx, bracket in enumerate(brackets)]
      bracket_id = sorted(bracket_budget, key=lambda x: x[1]*10 - x[0])
      print(bracket_id)
      bracket_id = bracket_id[0][0]
      trial = brackets[bracket_id].get_trial()
      if _get_metadata(trial, 'hyperband_id') < 0:
        # new trial
        parameters = self._delegate_policy.GetNewSuggestions(1, completed_trials, pending_trials)[0].parameters
        trial.parameters = parameters
        _set_metadata(trial, 'hyperband_id', next_hyperband_id)
        next_hyperband_id += 1
      trials.append(trial)
    return trials

  def summary(self, completed_trials, pending_trials):
    brackets, next_hyperband_id = self._construct_brackets(completed_trials, pending_trials)
    statement = '\n'
    statement += str([int(bracket.total_records) for idx, bracket in enumerate(brackets)])
    for idx, bracket in enumerate(brackets):
      statement +=  '\ns=%d\n' % idx
      statement += bracket.bracket_string()
    return statement

class SHAPolicy(object):
  def __init__(self, delegate_policy, max_records, eta=3, halving_rounds=3, goal_maximize=False):
    self._delegate_policy = delegate_policy
    self._eta = eta
    self._min_records = max_records*self._eta**(-halving_rounds)
    self._halving_rounds = halving_rounds
    self._goal_maximize = goal_maximize
    self._max_records = max_records

  def _construct_brackets(self, completed_trials, pending_trials):
    all_trials = set(completed_trials).union(set(pending_trials))
    next_hyperband_id = -1
    if len(all_trials) > 0:
      next_hyperband_id = max([_get_metadata(t, 'hyperband_id') for t in all_trials])
    next_hyperband_id += 1
    s = self._halving_rounds
    bracket = Bracket(self._min_records,
                        all_trials,
                        s,
                        self._eta,
                        s,
                        self._goal_maximize,
                        pending_trials)

    return bracket, next_hyperband_id

  def GetNewSuggestions(self, num_suggestions_hint, completed_trials, pending_trials):
    # brackets reconstructed each time, probably can modify with each new result
    bracket, next_hyperband_id = self._construct_brackets(completed_trials, pending_trials)
    trials = []
    for _ in range(num_suggestions_hint):
      trial = bracket.get_trial()
      if _get_metadata(trial, 'hyperband_id') < 0:
        # new trial
        parameters = self._delegate_policy.GetNewSuggestions(1, completed_trials, pending_trials)[0].parameters
        trial.parameters = parameters
        _set_metadata(trial, 'hyperband_id', next_hyperband_id)
        #print(next_hyperband_id, _get_metadata(trial,'termination_record'))
        next_hyperband_id += 1
      trials.append(trial)
    return trials

  def summary(self, completed_trials, pending_trials):
    bracket, next_hyperband_id = self._construct_brackets(completed_trials, pending_trials)
    statement = '\n'
    statement += 'total records: %d' % int(bracket.total_records)
    statement +=  '\ns=%d\n' % self._halving_rounds
    statement += bracket.bracket_string()
    return statement


