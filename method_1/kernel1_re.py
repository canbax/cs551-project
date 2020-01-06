# Yusuf Sait Canbaz
# thanks to https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
import numpy as np
import pandas as pd
import time

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os


def trainsform_features(isConvert2Numeric=True):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Now drop the  'Id'
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)

    # Deleting outliers
    train = train[train.GrLivArea < 4500]
    train.reset_index(drop=True, inplace=True)

    # log transform to see more normal distribution
    train['SalePrice'] = np.log1p(train['SalePrice'])
    y = train.SalePrice.reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test

    features = pd.concat([train_features, test_features]
                         ).reset_index(drop=True)

    # some numeric features are actually categorical features
    features['MSSubClass'] = features['MSSubClass'].astype(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)

    # fill default values for some non-existing
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna('SBrkr')
    features['KitchenQual'] = features['KitchenQual'].fillna('TA')
    features['Exterior1st'] = features['Exterior1st'].fillna(
        features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(
        features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(
        features['SaleType'].mode()[0])

    features['PoolQC'] = features['PoolQC'].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    features['MSZoning'] = features.groupby(
        'MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)

    features.update(features[objects].fillna('None'))

    features['LotFrontage'] = features.groupby(
        'Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Filling in the rest of the NA's

    numeric_dtypes = ['int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    features.update(features[numerics].fillna(0))

    numeric_dtypes = ['int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
    numerics2 = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics2.append(i)

    skew_features = features[numerics2].apply(
        lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))

    features = features.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)

    final_features = features
    if isConvert2Numeric:
        final_features = pd.get_dummies(features).reset_index(drop=True)

    X = final_features.iloc[:len(y), :]
    X_test = final_features.iloc[len(X):, :]

    outliers = [30, 88, 462, 631, 1322]
    X = X.drop(X.index[outliers])
    y = y.drop(y.index[outliers])

    # some dummy features might occur like an ID which is unique for each sample
    # delete them if they exists
    too_unique_features = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(X) * 100 > 99.94:
            too_unique_features.append(i)

    too_unique_features = list(too_unique_features)
    # if we convert such a feature occurs
    if isConvert2Numeric:
        too_unique_features.append('MSZoning_C (all)')

    X = X.drop(too_unique_features, axis=1).copy()
    X_test = X_test.drop(too_unique_features, axis=1).copy()
    return X, X_test, y


X, _, y = trainsform_features()
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# build our model scoring function


def cv_rmsle(model, X=X):
    rmsle = np.sqrt(-cross_val_score(model, X, y,
                                     scoring='neg_mean_squared_log_error',
                                     cv=kfolds))
    return rmsle


svr = make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003,))


gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=42)


xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

# stack
stack_gen = StackingCVRegressor(regressors=(gbr, svr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


print('TEST score on CV')

start_time = time.time()
score = cv_rmsle(svr)
print('SVR rmsle: ', score.mean(), ' std: ', score.std())
print(time.time() - start_time, ' passed ')