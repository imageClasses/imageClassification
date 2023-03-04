"Loading data in, data wrangling, and so on"

import os
import numpy as np
import pandas as pd

data_folder = "data"

def split_img_indices(split=None):
    "Returns index lists based on split parameter"
    data_size = get_data_size()
    split = np.array(split)

    #Normalize to 1 just to be sure
    split /= split.sum()

    to_list = np.random.choice(len(split), size=data_size, p=split)

    lists = [[] for l in range(len(split))]
    for i in range(1, data_size+1):
        lists[to_list[i-1]].append(i)
    
    return tuple(lists)

def get_one_hot_df(indices=None):
    "Returns a dataframe with img indices and one-hotted labels. If no indices returns full data"
    label_indices = get_label_indices()
    if not indices:
        indices = list(range(1, get_data_size()+1))
    else:
        #Filter out label indices not in df
        for label, idx_list in label_indices.items():
            label_indices[label] = [idx for idx in idx_list if idx in indices]

    df = pd.DataFrame(0, index=indices, columns=get_labels())
    
    for label, indices in label_indices.items():
        df.loc[indices, label] = 1

    return df

def get_target_dfs(train=1.0, test=None, val=None):
    "Returns one-hotted target dfs with parametrized proportions"
    proportions = []
    if train == 1.0:
        return get_one_hot_df()
    else:
        proportions.append(train)
        if test is None and val is None:
            the_rest = 1.0 - train
            proportions.append(the_rest)
        else:
            if test:
                proportions.append(test)
            if val:
                proportions.append(val)
    

    idx_set = split_img_indices(proportions)
    df = get_one_hot_df()
    dfs = []
    for indices in idx_set:
        dfs.append(df.loc[indices])
    
    return tuple(dfs)


def get_label_indices(label=None):
    "Gets list of img indices tagged with label. If no label, gets everything"
    if label:
        return _read_label_indices(f"{label}.txt")
    else:
        label_indices = {}
        for label_file in os.listdir(f'{data_folder}/annotations'):
            label_indices[label_file.split(".")[0]] = _read_label_indices(label_file)
        return label_indices


def _read_label_indices(label_file):
    with open(f"{data_folder}/annotations/{label_file}", "r") as f:
        indices = [int(line) for line in f.readlines()]
    return indices



def get_labels():
    "Returns a list of label names"
    labels = os.listdir(f'{data_folder}/annotations')
    labels = [label.split(".")[0] for label in labels]
    return labels

def get_data_size():
    "Counts the number of images"
    return len(os.listdir(f"{data_folder}/images"))



