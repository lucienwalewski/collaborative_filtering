import pandas as pd

def submit(preds):

    # Load the sample submission file
    submission = pd.read_csv('../data/sampleSubmission.csv')

    # Extract the row and column indices from the 'Id' column
    submission['row_idx'] = submission['Id'].str.extract(r'r(\d+)_c\d+').astype(int) - 1
    submission['col_idx'] = submission['Id'].str.extract(r'r\d+_c(\d+)').astype(int) - 1

    # Update the predictions in the submission file
    submission['Prediction'] = preds[submission['row_idx'], submission['col_idx']]

    # Drop the row and column indices
    submission.drop(['row_idx', 'col_idx'], axis=1, inplace=True)

    # Save the updated submission file
    submission.to_csv('../data/newSubmission.csv', index=False)