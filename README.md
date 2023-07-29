# Project: Collaborative Filtering

This repository contains the code for the collaborative filtering project of the Computational Intelligence Lab course at ETH Zurich.
It implements different models for the task of predicting the rating of a user for a movie. The models are:
- Alternating Least Squares (ALS)
- K-Nearest Neighbors (KNN)
- Multi Layer Perceptron (MLP)
- Bayesian Factorization Machine (BFM)
- Bayesian Factorization Machine with additional features (BFM+)
- Ensemble model

The notebooks give an overview of the results obtained with each model. More specifically, the notebook `ensemble.ipynb` shows the results obtained with the ensemble model, used for the final submission. The folders `als` and `mlp` enabled us to tune these models.


# Installation
- conda env create -f environment.yaml

# Usage
- conda activate collaborative_filtering
- The folder containing all the data (which can be found here https://polybox.ethz.ch/index.php/s/tFQYOGEcoEQeeJb) should be placed in the root directory of the project for the notebooks to work.

# Structure of the code
```
├── als               <- ALS model
    ├── data_processing  <- Data processing scripts
    ├── models           <- Trained and serialized models, model predictions, or model summaries
    ├── pipeline        <- Pipeline scripts to Submit and Train the model
    ├── submission      <- Submission scripts
    ├── submission_task.py <- Submission script for the task to be launched
    ├── train_task.py  <- Training script for the task to be launched

├── data
    ├── data_train.csv <- Training data
    ├── mySubmission.csv <- Submission file
    ├── sampleSubmission.csv <- Sample submission file

├── mlp               <- MLP model
    ├── scripts  <- sbatch scripts to launch to submit
    ├── utils          <- Loading the dataset, pytorch dataset and model
    ├── train.py  <- Training script for the task to be launched
    ├── predict.py  <- Prediction task

├── notebooks         <- Jupyter notebooks. 
    ├── bayesian_fm.ipynb <- Bayesian FM model
    ├── bayesian_fm_plus.ipynb <- Bayesian FM model with additional features
    ├── bfm_active_learning.ipynb <- Bayesian FM model with active learning
    ├── ensemble.ipynb <- Ensemble model
    ├── knn_baseline.ipynb <- KNN baseline model

├── environment.yaml  <- The requirements file for reproducing the analysis environment, e.g.
                         generated with `conda env export > environment.yaml`

├── README.md          <- The top-level README for developers using this project.

```

# Data

```

├── ensemble <- Predictions and features need for the ensemble model to work (of each model used in our analysis)

├── data
    ├── data_train.csv <- Training data
    ├── mySubmission.csv <- Submission file
    ├── sampleSubmission.csv <- Sample submission file

├── ALS  <- Matrices saved in numpy format computed from the ALS model
```

