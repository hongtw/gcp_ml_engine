# from xgboost import XGBClassifier
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def build_estimator(config):
    params = config.get('params')
    return RandomForestClassifier(**params)