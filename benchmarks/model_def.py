import abc
class BenchmarkDef(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def requires_R(self):
        return
    @abc.abstractmethod
    def requires_device(self):
        return
    @abc.abstractmethod
    def get_search_space(self):
        return
    @abc.abstractmethod
    def get_allowed_datasets(self):
        return
    @abc.abstractmethod
    def model_params(self):
        return
    def set_seed(self,seed):
        self.seed =  seed
    def get_seed(self):
        return self.seed 
    @abc.abstractmethod
    def create_arm(self,dir,params,combined_train=False,default=False):
        """
        :param dir: output directory for generated arms
        :param params: value of hps for arm to create
        :param combined_train: whether to train on combined training and validation set for this arm
        :param default: whether to fill in with default values for this arm
        :return: a dictionary with filled in hp values representing an arm
        """
        return
    @abc.abstractmethod
    def run_solver(self,arm,n_units):
        return

    def generate_arms(self,start_ind, n,dir,combined_train = False):
        arms={}
        for i in range(n):
            params = {}
            search_space = self.get_search_space()
            for hp in search_space.keys():
                params[hp] = search_space[hp].get_param_range(1,stochastic=True)[0]
            arm = self.create_arm(dir,params,combined_train=combined_train)
            arms[start_ind + i] = arm
        return arms
