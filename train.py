import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from TaiChiACVP.utils.std_logger import Logger
from TaiChiACVP.features.loadpcxa_org_v5 import load_and_dilation
from TaiChiACVP.classification.ArgsClassify import * 
from TaiChiACVP.classification.imbalanced_classification import *
from TaiChiACVP.visual.plot import plot_trees
import torch
import ipdb
@hydra.main(config_path="configs", config_name="train_org.yaml")
def main(cfg: DictConfig):
    # ipdb.set_trace()
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0
    features_categories = []
    if cfg.features.peptide.HC:
        HC_training_sets, HC_test_sets = {}, {}
        HC_training_labels, HC_test_labels = {}, {}
        # ipdb.set_trace()
        
        features_categories.extend(['AAC', 'DiC', 'gap', 'Xc', 'PHYC'])
        for nlab in cfg.data.files_list: 

            HC_train = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-train.csv",index_col=0)
            HC_test = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-test.csv",index_col=0)
            
            features_inds = HC_train.columns[2:-2]
            # ipdb.set_trace()
            HC_training_sets[nlab] = HC_train[features_inds].values
            HC_test_sets[nlab] = HC_test[features_inds].values
            HC_training_labels[nlab] = HC_train['label'].values
            HC_test_labels[nlab] = HC_test['label'].values
            
    if cfg.features.peptide.LLM:
        training_sets, test_sets,training_labels, test_labels = load_and_dilation(cfg)
        features_categories.extend([cfg.features.pretrained.dilation.type])
        for k,features in zip(cfg.features.pretrained.dilation.topK, cfg.features.pretrained.dilation.features_list):
            # ipdb.set_trace()
            all_perfs = {}
            for nlab in cfg.data.files_list:
    
                nlab_dir = nlab
                os.makedirs(nlab_dir,exist_ok=True)
                
                all_perfs[nlab]=pd.DataFrame()
                for layer in range(len(training_sets[features][nlab])):
                    
                    X_train_list, X_test_list = [], []
                    name = ''
                    features_names = []
                    for model in  cfg.features.peptide.keys():
    
                        if model == 'LLM' and cfg.features.peptide[model]:
    
                            Logger.info(f"Add {cfg.features.pretrained.dilation.type}...")
                            name += f'{cfg.features.pretrained.dilation.type}-{layer}'
                            
                            X_train_list.append(training_sets[features][nlab][layer])
                            X_test_list.append(test_sets[features][nlab][layer])
                            
                            features_names.extend([ f'{cfg.features.pretrained.dilation.type}{layer}-{i}d' for i in range(training_sets[features][nlab][layer].shape[1])])
    
                        if model == 'HC' and cfg.features.peptide[model]:
    
                            Logger.info("Add HC...")
                            name +='|HC'
                            X_train_list.append(HC_training_sets[nlab])
                            X_test_list.append(HC_test_sets[nlab])
    
                            features_names.extend(features_inds.tolist())
                    X_train = np.concatenate(X_train_list,1)
                    X_test = np.concatenate(X_test_list,1)
                    y_train = training_labels[features][nlab]# .to_numpy()
                    y_test = test_labels[features][nlab]# .to_numpy()
                    
                    
                    for mdl_name, mdl in model_dicts.items():
                        mdl_dir = os.path.join(nlab_dir, mdl_name)
                        os.makedirs(mdl_dir,exist_ok=True)
                        clf = mdl['model']  # For sklearn, clf.fit() will override the previous parameters
                        clf_params = mdl['param_grid']
                        # ipdb.set_trace()
                        if 'imblearn.ensemble' not in clf.__module__:
                            prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                                        imb_samplers=imb_strategies, estimator=clf,logger=Logger,
                                                                                        grid_params=clf_params, iteration=5)
                        else:
                            prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                                        imb_samplers={"Default": None},
                                                                                        estimator=clf,logger=Logger,
                                                                                        grid_params=clf_params, iteration=5)
                        
                        for spl_name in feature_importance.keys():
                            spl_dir = os.path.join(mdl_dir, spl_name)
                            os.makedirs(spl_dir,exist_ok=True)
                            
                            # plot trees
                            if cfg.plot.tree:                       
                              plot_trees(feature_names=features_names,
                                          interested_feature_name=cfg.features.pretrained.dilation.type,
                                          model=estms[spl_name],
                                          savedir=nlab_dir,
                                          name = f"{spl_dir}-{cfg.features.pretrained.dilation.type}-{layer}")
                            
                            # save feature importances    
                            feature_importance_df = pd.DataFrame.from_dict(dict(zip(features_names,feature_importance[spl_name])),orient='index')
                            
                            feature_importance_df.to_csv(os.path.join(spl_dir,f'feature_importance-{features}-{cfg.features.pretrained.dilation.type}{layer}.csv'))
                            
                            feature_importance_ratio = {f:0 for f in features_categories}
                            feature_number = {f: 0 for f in features_categories}
                            subsum = 0
                            fnum = 0
                            for fname in features_categories:
                                for idx, f_name in enumerate(features_names):
                                    if fname in f_name:
                                        feature_importance_ratio[fname] += feature_importance[spl_name][idx]
                                        feature_number[fname] += 1
                                        subsum += feature_importance[spl_name][idx]
                                        fnum += 1
                            
                            if cfg.features.peptide.HC:
                                # calculate PHYC
                                feature_importance_ratio[features_categories[-2]] = sum(feature_importance[spl_name]) - subsum
                                feature_number[features_categories[-2]] = len(features_names) - fnum
                                # ipdb.set_trace()
                                
                            feature_importance_ratio_df = pd.DataFrame([feature_importance_ratio,feature_number])
                            Logger.info(feature_importance_df)
                            feature_importance_ratio_df.to_csv(os.path.join(spl_dir,f'feature_importance_ratio-{features}-{cfg.features.pretrained.dilation.type}{layer}.csv'),index=False)
                        # Save the estimators, parameters, roc/pr results
                        
                        # perfs.loc[:,'features-layer'] = f'{features}-{layer}'
                        all_perfs[nlab] = all_perfs[nlab].append(perfs)
                      # ipdb.set_trace()
                all_perfs[nlab].to_csv(os.path.join(nlab_dir, f"{nlab}-{features}-{cfg.features.pretrained.dilation.type}-performances.csv"), index=False)
                
    else:
    
        for nlab in cfg.data.files_list:
          nlab_dir = nlab
          os.makedirs(nlab_dir,exist_ok=True)          
                    
          Logger.info("Only HC...")
          
          name ='HC'
          X_train=HC_training_sets[nlab]
          X_test=HC_test_sets[nlab]
          y_train = HC_training_labels[nlab]# .to_numpy()
          y_test = HC_training_labels[nlab]# .to_numpy()
          
          features_names = features_inds.tolist()
          
          for mdl_name, mdl in model_dicts.items():
              mdl_dir = os.path.join(nlab_dir, mdl_name)
              os.makedirs(mdl_dir,exist_ok=True)
              clf = mdl['model']  # For sklearn, clf.fit() will override the previous parameters
              clf_params = mdl['param_grid']
              # ipdb.set_trace()
              if 'imblearn.ensemble' not in clf.__module__:
                  prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                              imb_samplers=imb_strategies, estimator=clf,logger=Logger,
                                                                              grid_params=clf_params, iteration=5)
              else:
                  prcs, rocs, perfs, estms, params_estm,feature_importance = imb_classification(X_train, y_train, X_test, y_test,mdl_name,name,
                                                                              imb_samplers={"Default": None},
                                                                              estimator=clf,logger=Logger,
                                                                              grid_params=clf_params, iteration=5)
              
              for spl_name in feature_importance.keys():
                  spl_dir = os.path.join(mdl_dir, spl_name)
                  os.makedirs(spl_dir,exist_ok=True)
                  
                  # plot trees
                  if cfg.plot.tree:                       
                    plot_trees(feature_names=features_names,
                                interested_feature_name=None,
                                model=estms[spl_name],
                                savedir=nlab_dir,
                                name = f"{spl_dir}-HC")
                  
                  # save feature importances    
                  feature_importance_df = pd.DataFrame.from_dict(dict(zip(features_names,feature_importance[spl_name])),orient='index')
                  
                  feature_importance_df.to_csv(os.path.join(spl_dir,'feature_importance.csv'))
                  
                  feature_importance_ratio = {f:0 for f in features_categories}
                  feature_number = {f: 0 for f in features_categories}
                  subsum = 0
                  fnum = 0
                  for fname in features_categories:
                      for idx, f_name in enumerate(features_names):
                          if fname in f_name:
                              feature_importance_ratio[fname] += feature_importance[spl_name][idx]
                              feature_number[fname] += 1
                              subsum += feature_importance[spl_name][idx]
                              fnum += 1
                  
                  
                  # calculate PHYC
                  feature_importance_ratio[features_categories[-2]] = sum(feature_importance[spl_name]) - subsum
                  feature_number[features_categories[-2]] = len(features_names) - fnum
                  # ipdb.set_trace()
                      
                  feature_importance_ratio_df = pd.DataFrame([feature_importance_ratio,feature_number])
                  Logger.info(feature_importance_df)
                  feature_importance_ratio_df.to_csv(os.path.join(spl_dir,f'feature_importance_ratio-HC.csv'),index=False)
  

if __name__=='__main__':
    main()
    print('Done.')


    