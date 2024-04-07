import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# print(sys.path)
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModel,AutoConfig# ,LlamaModel
from TaiChiACVP.utils.std_logger import Logger
from TaiChiACVP.utils.scheduler import lr_scheduler
from TaiChiACVP.utils.distribution import setup_multinodes, cleanup_multinodes
from TaiChiACVP.loader.utils import make_loader
from TaiChiACVP.utils.utils import get_device,fix_random_seed
# from TaiChiACVP.features.query_features import FeatureFetcher
from TaiChiACVP.features.load_org import load_llm
# from TaiChiACVP.classification.classification import Classification
from TaiChiACVP.classification.ArgsClassify import * 
from TaiChiACVP.classification.imbalanced_classification import *
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.decomposition import PCA
import torch
import ipdb

@hydra.main(config_path="configs", config_name="train_org.yaml")
def main(cfg: DictConfig):
    # ipdb.set_trace()
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0

    # features_list = ['bert-base','bert-large']
    # cls_categories = ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"]
    
    training_sets, test_sets,training_labels, test_labels = load_llm(cfg)
    pca=None
    if cfg.features.pretrained.dilation.comparison: 
      if cfg.features.pretrained.dilation.percentage > 0:
      
        if cfg.features.pretrained.dilation.percentage < 1:
          pca = PCA(n_components=cfg.features.pretrained.dilation.percentage)
        else:
          pca = PCA()     
    else:
      pca = PCA(n_components=0.95)
    
    # ipdb.set_trace()
    for features in cfg.features.pretrained.dilation.features_list:
    
        all_perfs = {}
  
        for nlab in cfg.data.files_list:

            nlab_dir = nlab
            os.makedirs(nlab_dir,exist_ok=True)

            all_perfs[nlab]=pd.DataFrame()
            for layer in range(training_sets[features][nlab].shape[0]):
                # if layer > 1: break
                name = f'{cfg.features.pretrained.dilation.features_list[0]}a-{layer}'
                
                if pca is not None:
                  X_train = pca.fit_transform(training_sets[features][nlab][layer])# .to_numpy()
                  X_test = pca.transform(test_sets[features][nlab][layer])# .to_numpy()
                else:
                  X_train = training_sets[features][nlab][layer] # .to_numpy()
                  X_test = test_sets[features][nlab][layer] # .to_numpy()
                  
                y_train = training_labels[features][nlab]# .to_numpy()
                y_test = test_labels[features][nlab]# .to_numpy()
                # ipdb.set_trace()
                for mdl_name, mdl in model_dicts.items():
                    # mdl_dir = os.path.join(nlab_dir, mdl_name)
                    # os.makedirs(mdl_dir,exist_ok=True)
                    clf = mdl['model']  # For sklearn, clf.fit() will override the previous parameters
                    clf_params = mdl['param_grid']
                    if 'imblearn.ensemble' not in clf.__module__:
                        prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                                    imb_samplers=imb_strategies, estimator=clf,logger=Logger,
                                                                                    grid_params=clf_params, iteration=5)
                    else:
                        prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                                    imb_samplers={"Default": None},
                                                                                    estimator=clf,logger=Logger,
                                                                                    grid_params=clf_params, iteration=5)
                    # Save the estimators, parameters, roc/pr results
                    
                    # perfs.loc[:,'features-layer'] = f'{features}-{layer}'
                    all_perfs[nlab] = all_perfs[nlab].append(perfs)
                    # ipdb.set_trace()
            all_perfs[nlab].to_csv(os.path.join(nlab_dir, f"{nlab}-{features}-performances.csv"), index=False)
    # training_sets['hand_crafted'] = {

    #     nlab: pd.read_csv(f"/data/home/scv6872/PCBERTA/data/{nlab}-train.csv",index_col=0).iloc[:,2:-2].values
    # }   

    # test_sets['hand_crafted'] = {

    #     nlab: pd.read_csv(f"/data/home/scv6872/PCBERTA/data/{nlab}-test.csv",index_col=0).iloc[:,2:-2].values
    # }   

if __name__=='__main__':
    main()
    print('Done.')


    