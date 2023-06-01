import pandas as pd
from sklearn.model_selection import train_test_split


def load_cil(dataset="split"):

    file = "data_train" if dataset != "test" else "sampleSubmission"
    data = pd.read_csv(f'data/{file}.csv', index_col=0)
    data['user'] = data.index.str.split('_').str[0].str[1:].astype('int32')
    data['movie'] = data.index.str.split('_').str[1].str[1:].astype('int32')
    data.rename(columns={'Prediction': 'rating'}, inplace=True)
    data['rating'] = data['rating'].astype('uint8')
    data = data[['user', 'movie', 'rating']]

    print("Subtracted {} from user, movie and rating".format(1))
    data = data - 1  # make sure to undo for final file
    user_num = 10000  # int(data['user'].max() + 1)
    movie_num = 1000  # int(data['movie'].max() + 1)
    print("User num: {}, Movie num: {}".format(user_num, movie_num))

    train_data = val_data = None
    if dataset == "test":
        val_data = data
    elif dataset == "train":
        train_data = data
    else:
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data, val_data, user_num, movie_num


