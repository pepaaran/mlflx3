# A template for a data science project using python

This template is inspired by the 
[cookiecutter data science template](https://drivendata.github.io/cookiecutter-data-science/).

## Setting up the repository

Run the following code from the terminal to set up your repository and conda environment. 
Make sure that [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) 
and [python](https://wiki.python.org/moin/BeginnersGuide/Download) are installed prior. 
```
git clone git@github.com:pepaaran/python_proj_template.git
cd python_proj_template
conda env create -f environment.yml --name myenv
conda activate myenv
```
The `environment.yml` file in this template includes basic data science packages. When starting a new project,
you may want to assign a custom name to the environment, install any remaining packages used during
the project and update the environment file running:

```
conda env export --name myname
```

The previous code will update the packages specified in `environment.yml` to match all the
packages installed in your current conda environment, with their versions. This is key to a reproducible
workflow, so make sure that your environment file stays up to date with the packages (and versions)
used in your project.

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

Here you should provide an example for how to run your code, from beginning to end, script
by script. The more detailed (yet straight to the point) you are, the happier your future self
and all your collaborators will be.

Data should be kept outside of your repository, whenever it is too big to fit into GitHub
or there are privacy concerns. In these cases, give explanations of where users can download
or obtain the data and where they should save it, such that the whole workflow runs smoothly.
