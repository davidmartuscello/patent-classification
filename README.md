# Patent Classification

## Motivation

This repository acts as a proof of concept for patent application classification.


## Instructions to Run

pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchtext
pip install bayesian-optimization
python main.py

#### To Run on NYU HPC Cluster
1) create virtual environment at ~/project3/torchtext_env/
2) install required program
2) >> sbatch model_experiment
