import os
import pandas as pd
import numpy as np
from collections import Counter

def clear_binary_data(all_data: pd.DataFrame) -> pd.DataFrame:
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)

    sequence = all_data["Sequence"].values.tolist()
    seqName = all_data["ID"].values.tolist()
    labels =  all_data["label"].values.tolist()
    seqID = all_data["Idx"].values.tolist()
    
    label_dict = {"binary": labels}
    
    return sequence, seqName, seqID, label_dict


