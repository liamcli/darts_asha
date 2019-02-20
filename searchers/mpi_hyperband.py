#!/usr/bin/env python
"""
MPI wrapper for ASHA and asynchronous Hyperband.

MPI code based off of that by Craig Finch (cfinch@ieee.org).
Inspired by http://math.acadiau.ca/ACMMaC/Rmpi/index.html
"""
import getopt
import os
import pickle
import sys
import time

from mpi4py import MPI
from searchers.parallel.hyperband_policy import *
from searchers.parallel.hp_utils import *
import logging

def main(argv):

    model='darts_ptb'
    #input_dir='/home/ubuntu/data'
    data_name='ptb'
    output_dir='fill_in'
    seed=0
    algo = 'asha'
    run_time = 24*7*3600
    halving_rounds=4
    max_records = 256
    try:
        opts, args = getopt.getopt(argv,"ha:m:d:o:s:r:R:t:",['algo=', 'model=','data_name=','output_dir=','seed=','halving_rounds=','max_records=','time='])
    except getopt.GetoptError:
        print('mpi_continuous_hyperband.py -a <algo> -m <model> -d <data_name> -o <output_dir> -s <rng_seed> -r <halving_rounds> -R <max records> -t <time  in hours>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('mpi_continuous_hyperband.py -a <algo> -m <model> -d <data_name> -o <output_dir> -s <rng_seed> -r <halving_rounds> -R <max records> -t <time  in hours>')
            sys.exit()
        elif opt in ("-a", "--algo"):
            algo = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-d", "--data_name"):
            data_name = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-r", "--halving_rounds"):
            halving_rounds = int(arg)
        elif opt in ("-R", "--max_records"):
            max_records = int(arg)
        elif opt in ("-t", "--time"):
            run_time = int(arg)*3600

    def enum(*sequential, **named):
        """Handy way to fake an enumerated type in Python
        http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
        """
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    save_interval = 3600/12
    save_time = 3600/12
    start_time = time.time()

    output_dir=output_dir+'/'+model
    output_dir += '/'+data_name+'/bracket'+str(halving_rounds)+ '/trial'+str(seed)

    resume = os.path.exists(os.path.join(output_dir, 'cont_hyperband.pkl'))



    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        setup_logger(output_dir)
        # Master process executes code below
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        closed_workers = 0
        num_workers = size - 1
        def duration():
            return time.time() - start_time
        model = Model_Job(model,data_name,output_dir,seed,max_records,0)
        delegate_policy = RandomSamplingPolicy(model.arm_generator, max_records)

        if algo=='hyperband':
            policy = HyperbandPolicy(delegate_policy, max_records=max_records, eta=4, max_halving_rounds=halving_rounds, goal_maximize=True)
        elif algo=='asha':
            policy = SHAPolicy(delegate_policy, max_records=max_records, eta=4, halving_rounds=halving_rounds, goal_maximize=True)
        else:
            raise Exception('Selected hyperparameter optimization algorithm not valid.  Choose between "asha" or "hyperband"')

        conductor = Conductor(policy)
        if resume:
            logging.info('Resuming from previously saved file...')
            conductor.resume(os.path.join(output_dir, 'cont_hyperband.pkl'))
            logging.info(conductor.policy.summary(conductor.completed_trials.values(), conductor.pending_trials.values()))
        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if duration() < run_time:
                    trial_start = time.time()
                    trial = conductor.get_suggestion()
                    trial.timing['start_time'] = trial_start
                    comm.send(trial, dest=source, tag=tags.START)
                    logging.info("Sending configuration %d to worker %d with termination record %d" % (trial.metadata['hyperband_id'], source,trial.metadata['termination_record']))
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                logging.info("Remaining time is %0.2f" % ((run_time - duration())/60.0))
                trial = data
                logging.info("Received configuration %d with objective %0.2f and test result %0.2f from worker %d" % (trial.metadata['hyperband_id'],trial.measurements[-1].objective_value,trial.measurements[-1].test_acc, source))
                conductor.report_done(trial)
                trial.timing['end_time'] = time.time()
                if duration() > save_time:
                    logging.info('Saved trials to pickle file')
                    save_time += save_interval
                    pickle.dump(conductor.completed_trials,open('cont_hyperband.pkl','wb'))
            elif tag == tags.EXIT:
                logging.info("Worker %d exited." % source)
                closed_workers += 1

        pickle.dump(conductor.completed_trials,open('cont_hyperband.pkl','wb'))
        logging.info("Master finishing")
    else:
        while not os.path.exists(output_dir):
            time.sleep(1)
        setup_logger(output_dir)

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        # Setting devices this way instead of via the deep learning framework is more robust.
        # For example, in pytorch, if trying to restore a checkpoint on a different GPU, an error may be raised.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank - 1)

        model = Model_Job(model,data_name,output_dir,seed,max_records,0)
        logging.info("I am a worker with rank %d on %s." % (rank, name))
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            trial = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:
                # Do the work here
                trial = model.train(trial,start_time,run_time)
                comm.send(trial, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
if __name__ == "__main__":
    main(sys.argv[1:])
