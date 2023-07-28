## collaborative_filtering

# Installation
- conda env create -f environment.yaml

# Usage
- conda activate collaborative_filtering

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

The folder "delivery", provided with the project, should be placed in the root of the project (at the same level of the collaborative_filtering folder). It contains the data used for the project.
```

├── ensemble <- Predictions and features need for the ensemble model to work (of each model used in our analysis)

├── data
    ├── data_train.csv <- Training data
    ├── mySubmission.csv <- Submission file
    ├── sampleSubmission.csv <- Sample submission file

├── ALS  <- Matrices saved in numpy format computed from the ALS model
```

