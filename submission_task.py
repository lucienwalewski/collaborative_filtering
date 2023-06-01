import numpy as  np 
from pipeline import SubmissionPipeline

# load P and Q from P.npy and Q.npy
if __name__ == "__main__":
    P = np.load("P.npy")
    Q = np.load("Q.npy")
    preds = np.dot(P.T, Q)
    SubmissionPipeline(preds).execute()