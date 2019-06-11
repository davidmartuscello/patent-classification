# Patent Classification

## Motivation

This repository acts as a proof of concept for patent application classification. The data is taken from https://www.uspto.gov/learning-and-resources/electronic-data-products/office-action-research-dataset-patents.


## Instructions to Run

#### To Run on NYU HPC Cluster:
*  Create virtual environment at ~/project3/torchtext_env/

*  Run command in terminal:
`cd ~/project3/torchtext_env/`
`git clone https://github.com/trueMastermind/patent-classification.git`
`module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2`
`source ~/project3/torchtext_env/py3.6.3/bin/activate`

*  Install required program:
`pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl`
`pip3 install torchtext`
`pip install bayesian-optimization`

* Get input data:
	* Put office_actions.csv under
	`~/project3/torchtext_env/patent-classification/data`
	* Put json files under
	`~/project3/torchtext_env/patent-classification/json_files`

* Train and Run model:
`cd ~/project3/torchtext_env/patent-classification/Text-Classification-Pytorch-master/`
	* Traning models and do hyper-parameters tuning:
	`sbatch parameters-tuning`
	Then you will see a file params-tuning.job_id containing the best hyper-parameters
	* Get a best model to predict: change the command in run_best_model with the best hyper-parameters you just got and run:
	`sbatch run_best_model`

## Other
The `Check_json_files` python notebook captures the population counts of the various classes inside the data. The `office_actions.csv` file has to be kept in the same folder as the script but the json files need to be present in another folder called `json_files_1` in the parent of the current directory.
