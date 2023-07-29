import numpy as  np 
from pipeline import SubmissionPipeline

# load P and Q from P.npy and Q.npy
if __name__ == "__main__":
    P = np.load("models/ALS_weights/U.npy")
    Q = np.load("models/ALS_weights/V.npy")
    preds = np.dot(P, Q.T)
    SubmissionPipeline(preds).execute()