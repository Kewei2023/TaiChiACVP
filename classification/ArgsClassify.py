from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import EasyEnsembleClassifier as EEC
from imblearn.ensemble import BalancedRandomForestClassifier as BRC
from imblearn.ensemble import RUSBoostClassifier as RUSBC
from imblearn.ensemble import BalancedBaggingClassifier  as BBC
from sklearn.svm import SVC
import numpy as np
SAMPLER_RNG = 114
CLASSIFIER_RNG = 514
SELECT_FEATURES =  True # whether to perform feature selection based on wilcoxon rank-sum
NORMALISE = False

## For models. You need to remove all imb_ensemble models while step-1 prediction training.
model_dicts = {

    "RF": {"model": RFC(random_state=CLASSIFIER_RNG), "param_grid": {"n_estimators": [50, 100, 120, 180, 200, 240]}},
    "BalancedRF": {"model": BRC(random_state=CLASSIFIER_RNG), "param_grid": {"n_estimators": [50, 100,120, 180, 200,240]}},

}


# For different imb samplers. remove all imb_samplers while step-1 prediction training.
imb_strategies = {
    # Under-sampling
    'NearMiss_3': NearMiss(version=3),
    # 'NearMiss_2': NearMiss(version=2),
    'Default': None
}
