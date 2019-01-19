from xgboost import XGBClassifier
import xgboost as xgb

def build_estimator(config):
    params = config.get('params')
    return XGBClassifier(**params)