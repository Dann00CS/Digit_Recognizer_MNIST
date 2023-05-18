import pandas as pd
import numpy as np
from tqdm import tqdm

def get_dataframe_with_complete_array(df):
    '''
    Transforms from the initial MNIST csv, with 784 columns, to a new Dataframe into an 28x28 array.
    Input:
    - df: Initial dataframe from data folder in Kaggle.
    Output:
    - Dataframe with "label" and "img_arr" features.
    '''
    train_dict = {"label": [], "img_arr": []}
    for _, row in tqdm(df.iterrows()):
        img_arr = np.zeros((28,28)).astype("int")
        for i in range(0,28):
            for j in range(0,28):
                img_arr[i,j] = row["pixel"+str(i*28+j)]
        train_dict["label"].append(row["label"]), train_dict["img_arr"].append(img_arr)
    return pd.DataFrame(train_dict)