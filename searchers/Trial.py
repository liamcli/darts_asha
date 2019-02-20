REQUESTED = 1
PENDING = 2
COMPLETE = 4

class Trial(object):
  def __init__(self):
    self.parameters = []
    self.measurements = []
    self.timing = {'start_time':0, 'end_time':0, 'comp_start':0,'comp_end':0}
    self.metadata = {}
    self.id = None
    self.status = REQUESTED
    self.trial_infeasible = False
