import pandas as pd
import numpy as np
import torch
import os
from .properties import Properties
import ipdb
from ..utils.std_logger import Logger

def load(cfg):
      # features_list = ['esm2_t36','esm2_t48','bert-base','bert-large']
      # cls_categories = ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"]
      Prop_test_sets = {}
      
      for nlab in cfg.data.files_list: 

            # Prop_train = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-train.csv",index_col=0)
            Prop_test = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-test.csv",index_col=0)
            
            
            # Prop_training_sets[features][nlab] = Properties(Prop_train)
            Prop_test_sets[nlab] = Properties(Prop_test)
      
      return Prop_test_sets

