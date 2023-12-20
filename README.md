# An implementation of GPP modelling with traditional and recursive neural networks

This repository is a refactoring of the [mlflx2](https://github.com/geco-bern/mlflx2) project.

## Setting up the repository

Run the following code from the terminal to set up your repository and conda environment. 
Make sure that [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) 
and [python](https://wiki.python.org/moin/BeginnersGuide/Download) are installed prior. 
```
git clone git@github.com:pepaaran/mlflx3.git
cd mlflx3
conda env create -f environment.yml --name mlflx3_env
conda activate mlflx3_env
```
The `environment.yml` file in this repository includes the packages necessary to run the python code to train the models on the GECO workstations. A second conda environment caleed `environment_cuda12.yml` was used to run the code on the GECO laptops, which have different GPUs and CUDA drivers versions. 

To work with PyTorch and train models on a GPU, make sure that your CUDA drivers are compatible with 
the `pytorch` version. Check how to install `pytorch` [here](https://pytorch.org/). Then you can check
if your setup works by starting `python` in a terminal and testing for the presence of accelerated GPU
deep learning capabilities with the following code snippet. The last command should return `True`!

```
import torch
torch.cuda.is_available()
```

NOTE: If you need to use a package that is unavailable via conda, install it with pip after you've created the 
conda environment. Do not play with conda again or you risk breaking your environment. Always write in the 
README.md of your repository in detail how to reproduce your environment, step by step.

The analysis of the GPP predictions is performed in R, borrowing from the previous [eval_mlflx.Rmd](https://github.com/geco-bern/mlflx2/blob/main/src/evaluation/eval_mlflx.Rmd). In order to ensure reproducibility, this repository is also an R project and the package versions used are saved in the `renv.lock` file. To use this, open the `mlflx3.Rproj` file in RStudio and load the necessary packages by running:
```
renv::restore()
```

## Directory structure

This is a general structure for a python project, which should be tailored to the needs of 
individual data analyses.

```
├── README.md          <- The top-level README for developers using this project.
├── LICENSE
|
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump. Never touch.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks for exploration only. Naming convention should contain
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-pap-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, bibliography (.bib)
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `conda env export > environment.yml`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
|
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module.
│   │
│   ├── data           <- Scripts to download or generate data.
│   │   └── 01_clean_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling.
│   │   └── 02_build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions.
│   │   ├── 03_train_model.py
│   │   └── 04_predict_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── 05_visualize.py
│
└── .gitignore         <- Indicates which files should be ignored when pushing.
```

## Running code

To reproduce the results used to study the modelling of GPP using LSTM and DNN,
you can execute the code in this repository in this workflow. 

> NOTE: The implementation allows for more flexibility, changing training parameters, dimension of the neural networks, etc. Check the scripts to explore the different possibilities.


The raw input data, obtained from FLUXNET2015 for a selection of sites, is 
contained in `data/raw/df_20210510.csv`. First, you will need to process these
data by removing non-relevant columns, imputing missing values of covariates and
GPP (using a simple KNN) and adding meta information about the sites.

```
cd src
python 00_preprocess_data.py
```

Next, you can train the LSTM and DNN models on the processed data, using leave-site-out
cross-validation to evaluate the model performance. The model
outputs will be saved in the `model` folder, including the model weights for 
the best epoch for each site (from the leave-site-out cross-validation) in 
`model/weights`, the training logs from tensorboard in `model/runs` and a csv
containing the predictions for each site in `model/preds`.

```
python 01_lstm_train_leave_site_out.py --n_epochs=150 --patience=20
python 02_dnn_train_leave_site_out.py --n_epochs=150 --patience=20
```


Here you should provide an example for how to run your code, from beginning to end, script
by script. The more detailed (yet straight to the point) you are, the happier your future self
and all your collaborators will be.

Data should be kept outside of your repository, whenever it is too big to fit into GitHub
or there are privacy concerns. In these cases, give explanations of where users can download
or obtain the data and where they should save it, such that the whole workflow runs smoothly.

### Tips for training the models on a remote server

The conda environment specified in `environment.yml` has been tested on the 
GECO workstations and should function, while `environment_cuda12.yml` is adapted
to the lab's laptops which have a different CUDA version.

In order to train the models on the workstations, without risking the training
being interrupted by, for example, an interrupted VPN connection, you may use
[nohup](). This makes the process run in the background and saves the standard
output in a text file.

```
cd src
nohup python 01_lstm_train_leave_site_out.py 1>&2 lstm_train.out &
```

It is also possible to supervise the model training remotely, running
TensorBoard as follows. You should run the following code from the terminal and
then navigate to [http://localhost:16006/](http://localhost:16006/).

```
# Start SSH connection, routed to a different port
ssh -L 16006:127.0.0.1:6006 username@ip_address  # for workstation2

# Move to project directory and activate conda environment
cd mlflx3
conda activate mlflx3_env    # tensorboard must be installed

# Launch tensorboard dashboard
tensorboard --logdir model/runs/lstm_lso_epochs_150_patience_20_hdim_256_conditional_0/
# change the runs folder to match the training output file name
```


