# Project: Collaborative Filtering

## Abstract
With the significant growth of available data relating to users' interests, recommendation systems have become increasingly valuable across domains. Here, we focus on the task of predicting the individual ratings of 10,000 users for 1,000 items. For this task, known as Collaborative Filtering with explicit feedback, a plethora of algorithms have been developed. While recent algorithms rely on deep learning techniques, some researchers question their competitiveness compared to older, more established models. Moreover, there is a lack of research focusing on aspects beyond performance, such as quantifying the certainty of the model's predictions. Here, we aim to close this gap. First, we train a wide variety of models and ensembles on our dataset, ranging from K-Nearest Neighbors (KNN) over Neural Collaborative Filtering (NCF) to Bayesian Factorization Machines (BFM). We compare the models not just in overall performance, but also investigate which aspects influence their performance most. Second, we leverage the probabilistic nature of the BFM to derive certainty estimates for each prediction. We show that these estimates can be used effectively in the context of Active Learning (AL). In this context, not all data is available from the beginning but sampled incrementally. Our proposed AL approach significantly outperforms the baseline in which additional data points are sampled randomly. Thereby, we can reduce the overall sampling cost in a real-world scenario.

## Overview

This repository contains the code for the collaborative filtering project of the Computational Intelligence Lab course at ETH Zurich.
It implements different models for the task of predicting the rating of a user for a movie. The models are:
- K-Nearest Neighbors (KNN)
- Alternating Least Squares (ALS)
- Matrix Factorization (MF)
- Multi Layer Perceptron (MLP)
- Neural Collaborative Filtering (NCF)
- Bayesian Factorization Machine (BFM)
- Bayesian Factorization Machine with additional features (BFM++flipped)
- Ensemble model

For reproduction of the final Kaggle Submission results refer to the notebook `ensemble.ipynb`. It is used to train the ensemble on the predictions of the following base models: KNN, MLP, NCF, BFM, BFM++flipped. The predictions of the base models are stored in the folder `ensemble` (see Section "Data"). To retrain the base models refer to the corresponding training scripts (see Section "Structure of the code").


## Installation
- conda env create -f environment.yaml
- conda activate collaborative_filtering

## Structure of the code
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

├── mlp               <- MLP, MF & NCF model
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

## Data
The folder containing all the data can be found [here](https://polybox.ethz.ch/index.php/s/tFQYOGEcoEQeeJb) and should be placed in the root directory of the project.

```

├── ensemble <- Predictions and features need for the ensemble model to work (of each model used in our analysis)

├── data
    ├── data_train.csv <- Training data
    ├── mySubmission.csv <- Submission file
    ├── sampleSubmission.csv <- Sample submission file

├── ALS  <- Matrices saved in numpy format computed from the ALS model
```

## Code Sources
Please note that we used the following code sources for this work:
- The MLP, MF and NCF models are an adaptation of the [implementation](https://github.com/guoyang9/NCF) by Yangyang Guo
- The BFM and BFM++flipped models rely on the [myFM library](https://myfm.readthedocs.io/en/stable/index.html) and its associated tutorials
- For the ALS model, the use of np.linalg.solve was inspired by [this blog post](https://github.com/benlindsay/als-wr-tutorial/blob/master/modified_notebook.ipynb)
