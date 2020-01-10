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


def delete_outliers_from_train(train):
    train = train[train.GrLivArea < 4500]
    train.reset_index(drop=True, inplace=True)
    return train


def convert2_categorical(x, cols):
    for col in cols:
        x[col] = x[col].astype(str)
    return x


def fill_missing_values(x):
    x['Functional'] = x['Functional'].fillna('Typ')
    x['Electrical'] = x['Electrical'].fillna('SBrkr')
    x['KitchenQual'] = x['KitchenQual'].fillna('TA')
    x['Exterior1st'] = x['Exterior1st'].fillna(
        x['Exterior1st'].mode()[0])
    x['Exterior2nd'] = x['Exterior2nd'].fillna(
        x['Exterior2nd'].mode()[0])
    x['SaleType'] = x['SaleType'].fillna(
        x['SaleType'].mode()[0])

    x['PoolQC'] = x['PoolQC'].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        x[col] = x[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        x[col] = x[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        x[col] = x[col].fillna('None')

    x['MSZoning'] = x.groupby(
        'MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    x.update(x.select_dtypes(['object']).fillna('None'))

    x['LotFrontage'] = x.groupby(
        'Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    return x


def fix_skewed(x, numeric_dtypes):
    skew_features = x.select_dtypes(numeric_dtypes).apply(
        lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        x[i] = boxcox1p(x[i], boxcox_normmax(x[i] + 1))
    
    return x


def drop_too_uniques(x, x_test, is_2_numeric):
    # some dummy features might occur like an ID which is unique for each sample
    # delete them if they exists
    too_unique_features = []
    for i in x.columns:
        counts = x[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(x) * 100 > 99.94:
            too_unique_features.append(i)

    too_unique_features = list(too_unique_features)
    # if we convert such a feature occurs
    if is_2_numeric:
        too_unique_features.append('MSZoning_C (all)')
    
    x = x.drop(too_unique_features, axis=1).copy()
    x_test = x_test.drop(too_unique_features, axis=1).copy()
    
    return x, x_test


def transform_features(is_delete_outlier_from_train=True, is_log_transform_y=True, is_2_numeric=True, is_2_categorical=True,
                       is_fill_missing=True, is_fill_missing_numeric=True, is_fix_skewed=True, is_drop_too_uniq=True):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Now drop the  'Id'
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)

    # Deleting outliers
    if is_delete_outlier_from_train:
        train = delete_outliers_from_train(train)

    # log transform to see more normal distribution
    if is_log_transform_y:
        train['SalePrice'] = np.log1p(train['SalePrice'])
    y = train.SalePrice.reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test

    features = pd.concat([train_features, test_features]
                         ).reset_index(drop=True)

    # some numeric features are actually categorical features
    if is_2_categorical:
        features = convert2_categorical(
            features, ['MSSubClass', 'YrSold', 'MoSold'])

    # fill missing values
    if is_fill_missing:
        features = fill_missing_values(features)

    # fill missing numerics
    numeric_dtypes = ['int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
    if is_fill_missing_numeric:
        features.update(features.select_dtypes(numeric_dtypes).fillna(0))

    if is_fix_skewed:
        fix_skewed(features, numeric_dtypes)
    
    final_features = features
    if is_2_numeric:
        final_features = pd.get_dummies(features).reset_index(drop=True)

    X = final_features.iloc[:len(y), :]
    X_test = final_features.iloc[len(X):, :]

    outliers = [30, 88, 462, 631, 1322]
    X = X.drop(X.index[outliers])
    y = y.drop(y.index[outliers])

    if is_drop_too_uniq:
        X, X_test = drop_too_uniques(X, X_test, is_2_numeric)
    
    return X, X_test, y


X, _, y = transform_features()
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


def cross_val_rmsle(model, X=X):
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
score = cross_val_rmsle(svr)
print('SVR rmsle: ', score.mean(), ' std: ', score.std())
print(time.time() - start_time, ' passed ')
