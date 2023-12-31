# An implementation of GPP modelling with traditional and recurrent neural networks

This repository is a refactoring of the [mlflx2](https://github.com/geco-bern/mlflx2) project.

#### Summary:

This repository implements a machine learning workflow to predict GPP. The models considered are a traditional deep neural network
(DNN), a recurrent neural network with an LSTM cell (Long-Short-Term-Memory) stacked with two extra connected layers, and a mixed model
that concatenates the LSTM output with categorical variables and then stacks connected layers on top (indicated by the conditional
argument in this implementation). The models are trained following two different workflows, to evaluate the out-of-sample performance:

- *Leave-site-out cross validation*: The whole time series from a single site is taken out as a test set. The remaining data is randomly split
by sites, stratified based on the mean temperature and aridity of the sites, with 20\% of the sites used for validation and 80\%
used for model training. The model is trained by minimizing the MSE for a maximum number of epochs and early stopping is used whenever the
MSE on the validation set has not improved for a given number of epochs (set by the patience parameter). The best model (the one with
the lowest MSE on the validation sites) is chosen and used to predict GPP on the left-out site (test data). Then the coefficient of 
determination R2 is computed. By repeating this workflow for each site in the dataset, we obtain an R2 score for each site and 
study the performance and generalization abilities of the machine learning models.

- *Leave-group-out cross validation*: The sites in the dataset are split into groups based on vegetation type or continent. For each group,
the model is trained with the leave-site-out procedure described above (hence there are less sites in the group and less data for training),
resulting in one model per site in the group (trained on the remaining sites in the group, with a train-validation split and early
stopping). To estimate the within-group and across-groups performance, we compute the bias of the GPP predictions (average difference
between observed and predicted values) for the left-out sites within the group - within-group bias - and for all the sites outside of 
the given group - across-group bias - averaged over all the trained models. 

The DNN, LSTM and conditional LSTM models are compared against each other, but also against the P-model (mechanistic photosyntehsis model).

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
│   ├── external       <- Data from third party sources, includes flux sites metadata.
│   ├── processed      <- The final data sets for modeling.
│   └── raw            <- The original, immutable data dump. Never touch.
│
├── models             <- Trained and serialized models (weights), model predictions, and trainings logs.
│
├── notebooks          <- Jupyter notebooks for exploration only. RMarkdowns for analysis of results.
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
│   ├── data           <- Functions to preprocess and format data for pytorch use.
│   │
│   ├── features     
│   │
│   ├── models         <- Functions to define model structures.
│   │
│   ├── visualization  
|   |
│   └── utils          <- General purpose functions and functions to train and test models (train-test split, loops, etc.).
│   │
│   ├── 00_preprocess_data.py
│   ├── 01_lstm_train_leave_site_out.py
│   ├── 02_dnn_train_leave_site_out.py
│   └── 03_lstm_train_leave_vegetation_out.py
|
├── R                  <- Contains custom R functions used in the results analysis.
│
├── .gitignore         <- Indicates which files should be ignored when pushing.
├── mlflx3.Rproj       <- Makes the repository an R project
├── renv.lock          <- Saves the package versions used in the R code
└── 
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

The code to perform the leave-vegetation-out and leave-continent-out cross-validation
is still being finished. Together with the saved objects as for the normal LSTM and DNN,
this analysis saves the mean bias for each site (models trained on different combinations
of data).
```
# Train LSTM model, leaving one vegetation type out at a time
python 03_lstm_train_leave_vegetation_out.py --group_name="DBF" --n_epochs=150 --patience=20
python 03_lstm_train_leave_vegetation_out.py --group_name="ENF" --n_epochs=150 --patience=20
python 03_lstm_train_leave_vegetation_out.py --group_name="GRA" --n_epochs=150 --patience=20
python 03_lstm_train_leave_vegetation_out.py --group_name="MF" --n_epochs=150 --patience=20
```

Finally, the analysis and visualization of results are implemented in an 
RMarkdown file in `notebooks/evaluate_outputs.Rmd`. This file reads the GPP
predictions from the `models/preds/` folder and produces a series of plots,
some of them used for the initial manuscript. To knit this report or run the
code chunks, you can open the file in RStudio and work inside the R Project.

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

# Launch tensorboard dashboard as usual
tensorboard --logdir model/runs/lstm_lso_epochs_150_patience_20_hdim_256_conditional_0/
# change the runs folder to match the training output file name
```


