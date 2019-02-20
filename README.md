This repo contains the code to run the experiments with ASHA on the two neural architecture search benchmarks presented in "Random Search and Reproducibility for NAS."  ASHA is short for the Asynchronous Successive Halving algorithm, which was presented in [this paper](https://arxiv.org/abs/1810.05934).


# ASHA
Please follow directions in [this repo](https://github.com/liamcli/randomNAS_release) to make sure you have the forked DARTS repo with the correct python packages and also the data downloaded into the right directories.

You will need the python package `mpi4py` in order to run ASHA in parallel.  

Make sure the path for `darts` and the data are correct in the benchmark files:  
`darts_asha/benchmarks/nas_search/cnn/darts/darts_wrapper.py`  
`darts_asha/benchmarks/nas_search/cnn/darts/darts_trainer.py`  
`darts_asha/benchmarks/nas_search/ptb/darts/darts_wrapper.py`  
`darts_asha/benchmarks/nas_search/ptb/darts/darts_trainer.py`  
Also, make sure `darts_asha` is in your `PYTHONPATH`.  

To run ASHA using multiple GPUs on the same machine, issue the following command from `darts_asha/searchers`:
`mpirun -np [# GPUs + 1] -output-filename [mpi filename root] python mpi_hyperband.py -a asha -m [darts_cnn/dart_ptb] -d [cifar10/ptb] -o [output dir] -s [seed] -r 1 -R 300 -t [# hours]`

The random seeds used for ASHA in the paper are available in [this spreadsheet](https://docs.google.com/spreadsheets/d/1XajrgOnNr7rST8sDYX8YVV_IHYlI98h21JRph0Uz6QU/edit?usp=sharing).  Note that this repo contains the fix for non-deterministic CNN training, while our experiments were conducted prior to this fix using the original random seeding in DARTS.
