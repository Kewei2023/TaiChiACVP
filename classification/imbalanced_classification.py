import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ranksums
import json
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from imblearn.metrics import geometric_mean_score
# from ArgsClassify import *  # Parameters for training classifier


def evaluate(X, y, estm):
    # Performance metrics
    y_pred = estm.predict(X)
    print(confusion_matrix(y, y_pred).ravel())
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # ROC curve
    try:
        if "decision_function" not in dir(estm):
            y_prob = estm.predict_proba(X)[:, 1]
        else:
            y_prob = estm.decision_function(X)
        pre, rec, _ = precision_recall_curve(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        aucroc = auc(fpr, tpr)
        aucpr = auc(rec, pre)
    except AttributeError:
        print("Classifier don't have predict_proba or decision_function, ignoring roc_curve.")
        pre, rec = None, None
        fpr, tpr = None, None
        aucroc = None
        aucpr = None
    eval_dictionary = {
        "CM": confusion_matrix(y, y_pred),  # Confusion matrix
        # "bACC": 0.5( tp / (tp + fp) + tn / (tn + fn)),
        "ACC": (tp + tn) / (tp + fp + fn + tn),  # accuracy
        "F1": fbeta_score(y, y_pred, beta=1),
        "F2": fbeta_score(y, y_pred, beta=2),
        "GMean": geometric_mean_score(y, y_pred, average='binary'),
        "SEN": tp / (tp + fn),
        "PREC": tp / (tp + fp),
        "SPEC": tn / (tn + fp),
        "MCC": matthews_corrcoef(y, y_pred),
        "PRCURVE": {"precision": pre, "recall": rec, "aucpr": aucpr},
        "ROCCURVE": {"fpr": fpr, "tpr": tpr, "aucroc": aucroc}
    }
    return eval_dictionary



def train_gridCV_imb(X, y, estm, sampler, param_grid, scoring='recall'):
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    grid = GridSearchCV(estm, param_grid,
                        scoring=scoring, cv=10, n_jobs=-1)  # use recall for finding results
    grid.fit(X_res, y_res)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, X_res, y_res

def show_other_idx(X_tr,y,estm,sampler,logger):
  skf = StratifiedKFold(n_splits=10)
  
  
  all_selected = {}
  sen_all,spec_all = [], []
  f1_all, f2_all, gmean_all = [], [], []
  mcc_all, aucpr_all, aucroc_all = [], [], []
  prec_all = []
  for train_index, test_index in skf.split(X_tr, y):
  
    X_tra, X_tes = X_tr[train_index], X_tr[test_index]
    y_tra, y_tes = y[train_index], y[test_index]
    
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_tra, y_tra)
    else:
        X_res, y_res = X_tra, y_tra
    
    y_pred = estm.predict(X_tes)
    eval_d = evaluate(X_tes, y_tes, estm)
    
    f1_all.append(eval_d['F1'])
    f2_all.append(eval_d['F2'])
    gmean_all.append(eval_d['GMean'])
    sen_all.append(eval_d['SEN'])
    prec_all.append(eval_d['PREC'])
    spec_all.append(eval_d['SPEC'])
    mcc_all.append(eval_d['MCC'])
    
    pr_cur = eval_d['PRCURVE']
    roc_cur = eval_d['ROCCURVE']
    aucpr_all.append(pr_cur['aucpr'])
    aucroc_all.append(roc_cur['aucroc'])
    
    
    
    
  gmean = np.array(gmean_all).mean(0)
  f1 = np.array(f1_all).mean(0)
  f2 = np.array(f2_all).mean(0)
  sen = np.array(sen_all).mean(0)
  prec = np.array(prec_all).mean(0)
  spec = np.array(spec_all).mean(0)
  mcc = np.array(mcc_all).mean(0)
  aucpr = np.array(aucpr_all).mean(0)
  aucroc = np.array(aucroc_all).mean(0)
  
  
  all_selected['gmean']= gmean
  all_selected['f1']= f1
  all_selected['f2']= f2
  all_selected['sen']= sen
  all_selected['prec'] = prec
  all_selected['spec']= spec
  all_selected['mcc']= mcc
  all_selected['aucpr']= aucpr
  all_selected['aucroc']= aucroc
  
  
  logger.info('gmean for {} :'.format(all_selected['gmean']))
  logger.info('f1 for {} :'.format(all_selected['f1']))
  logger.info('f2 for {} :'.format(all_selected['f2']))
  logger.info('sen for {} :'.format(all_selected['sen']))
  logger.info('prec for {} :'.format(all_selected['prec']))
  logger.info('spec for {} :'.format(all_selected['spec']))
  logger.info('mcc for {} :'.format(all_selected['mcc']))
  logger.info('aucpr for {} :'.format(all_selected['aucpr']))
  logger.info('aucroc for {} :'.format(all_selected['aucroc']))
  
  return all_selected

def imb_classification(X_tr, y_tr, X_te, y_te, mdl_name,layer_list, imb_samplers, estimator,logger,
                       grid_params=None, X_u=None, iteration=16, scoring='recall'):
    # Load data
    print("==============================================================")
    print(
        "Number of training samples: %d, Positive: %d, Negative: %d" % (len(y_tr), y_tr.sum(), len(y_tr) - y_tr.sum()))
    print("Number of test samples: %d, Positive: %d, Negative: %d" % (len(y_te), y_te.sum(), len(y_te) - y_te.sum()))

    # Imbalanced learning with different imbalance strategies
    perf_df = []
    FIM = {}
    pr_curves_all, roc_curves_all = {}, {}
    best_estimators_dict, best_params_dict = {}, {}
    for sampler_name, sampler in imb_samplers.items():
        # Train with sampled data
        acc_all, sen_all, prec_all, spec_all = [], [], [], []
        f1_all, f2_all, gmean_all = [], [], []
        mcc_all, aucpr_all, aucroc_all = [], [], []
        pr_cur, roc_cur = None, None  # Not support for mean PR-curve yet
        best_estimator, best_params = None, None
        # determine the iteration of sampler using randomized strategy
        if 'random_state' not in dir(sampler):
            iter_use = 1
        else:
            if sampler.random_state is not None:
                print("Perform single_iter sampling since random_state is set.")
                iter_use = 1
            else:
                iter_use = iteration
        print("Module of sampling strategy: {}, Evaluating with {:2d} iterations".format(sampler_name, iter_use))
        for ii in range(iter_use):
            if 'random_state' in dir(sampler):
                if sampler.random_state is None:
                    sampler.random_state = 0 + ii
            if X_u is None:
                estm, params, score, _, _ = train_gridCV_imb(X_tr, y_tr, estimator, sampler, grid_params, scoring=scoring)
                all_selected = show_other_idx(X_tr,y_tr,estm,sampler,logger)
            # For the TriTraining only
            # '''
            else:
                if sampler is not None:
                    X_r, y_r = sampler.fit_resample(X_tr, y_tr)
                else:
                    X_r, y_r = X_tr, y_tr
                # '''
                # insert RFECV
                # '''
                
                
                estimator.fit(X_r, y_r, X_u)
                estm = estimator
                params = None
            # '''
            
            
            FIM[sampler_name] = estm.feature_importances_
            eval_d = evaluate(X_te, y_te, estm)  # evaluate with test data
            
            # judge the best estimator
            if len(acc_all) == 0:
                best_estimator = estm
                best_params = params
            else:
                best_estimator = estm if eval_d['SEN'] > sen_all[-1] else best_estimator
                best_params = params if eval_d['SEN'] > sen_all[-1] else best_params
            acc_all.append(eval_d['ACC'])
            f1_all.append(eval_d['F1'])
            f2_all.append(eval_d['F2'])
            gmean_all.append(eval_d['GMean'])
            sen_all.append(eval_d['SEN'])
            prec_all.append(eval_d['PREC'])
            spec_all.append(eval_d['SPEC'])
            mcc_all.append(eval_d['MCC'])
            pr_cur = eval_d['PRCURVE']
            roc_cur = eval_d['ROCCURVE']
            aucpr_all.append(pr_cur['aucpr'])
            aucroc_all.append(roc_cur['aucroc'])
            

        perf_df.append({
            "Classifier": mdl_name,
            "Sampler": sampler_name,
            "Layer_list":layer_list,
            "CM": eval_d["CM"],
            "CV bACC(%)":"{}".format(score*100),
            
            "CV F1(%)": "{}".format(all_selected['f1'] * 100),
            "CV F2(%)": "{}".format(all_selected['f2'] * 100),
            "CV GMean(%)": "{}".format(all_selected['gmean'] * 100),
            "CV SEN(%)": "{}".format(all_selected['sen'] * 100),
            "CV PREC(%)": "{}".format(all_selected['prec'] * 100),
            "CV SPEC(%)": "{}".format(all_selected['spec'] * 100),
            "CV MCC(%)": "{}".format(all_selected['mcc'] * 100),
            "CV AUCPR(%)": ("{}".format(all_selected['aucpr'] * 100) if X_u is None else "Unavailable"),
            "CV AUCROC(%)": ("{}".format(all_selected['aucroc'] * 100) if X_u is None else "Unavailable"),
            
            "ACC(%)": "{}".format(np.mean(acc_all) * 100),
            "F1(%)": "{}".format(np.mean(f1_all) * 100),
            "F2(%)": "{}".format(np.mean(f2_all) * 100),
            "GMean(%)": "{}".format(np.mean(gmean_all) * 100),
            "SEN(%)": "{}".format(np.mean(sen_all) * 100),
            "PREC(%)": "{}".format(np.mean(prec_all) * 100),
            "SPEC(%)": "{}".format(np.mean(spec_all) * 100),
            "MCC(%)": "{}".format(np.mean(mcc_all) * 100),
            "AUCPR(%)": ("{}".format(np.mean(aucpr_all) * 100) if X_u is None else "Unavailable"),
            "AUCROC(%)": ("{}".format(np.mean(aucroc_all) * 100) if X_u is None else "Unavailable")
        })

        pr_curves_all[sampler_name] = pr_cur
        roc_curves_all[sampler_name] = roc_cur
        best_estimators_dict[sampler_name] = best_estimator
        best_params_dict[sampler_name] = best_params
    perf_df = pd.DataFrame(perf_df)
    logger.info("Table: Performance of selected samplers")
    logger.info(perf_df)
    return pr_curves_all, roc_curves_all, perf_df, best_estimators_dict, best_params_dict, FIM





        