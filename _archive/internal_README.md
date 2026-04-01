file structure of this repo:

job-scripts/ # bash scripts for launching jobs. most scripts that are launched are in /src/
src/ # stuff scripts in job-scripts/ calls
     models/ # model code
     analyses/ # analysis scripts
util/ # general purpose code you don't want to have to rewrite 1000 times
     plotting.py
     stats.py
     dataloaders.py
     etc
configs/ #run time parameter for things called in src/
figures/ # results graphs
docs/
ckpts/ # wont be pushed
notebooks/ # useful for testing out code. and then converting to scripts