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
    features.update(features.select_dtypes(numeric_dtypes).fillna(0))
    
    if is_fill_missing_numeric:
        features.update(features.select_dtypes(numeric_dtypes).fillna(features.mean()))
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


def rmsle(y, y_):
    """ A function to calculate Root Mean Squared Logarithmic Error (RMSLE) """
    assert len(y) == len(y_)
    return np.sqrt(np.mean((np.log(1 + y) - np.log(1 + y_))**2))


def cross_val_rmsle(model, X, y, is_logged_y=True, x2=[], y2=[], is_logged_y2=False):
    X = np.array(X)
    y = np.array(y)

    if len(x2) > 0:
        x2 = np.array(x2)
        y2 = np.array(y2)

    X = RobustScaler().fit_transform(X)
    num_split = 10
    kf = KFold(n_splits=num_split, shuffle=True)

    num_weight = 10
    scores = np.ones((num_split, ))
    cnt = 0
    for train_index, test_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        y_ = model.predict(X[test_index])
        if is_logged_y:
            y_ = np.exp(y_) - 1

        y_2 = 0
        if len(x2) > 0:
            y_2 = run_model2(x2[train_index], y2[train_index], x2[test_index])
            if is_logged_y2:
                y_2 = np.exp(y_2) - 1
        weight = 0
        y_comb = y_ * (1-weight) + weight * y_2
        # th = 0
        # cnt2 = 0
        # while cnt2 < num_weight:
        #     y_comb = y_ * (1-th) + th * y_2
        #     if is_log_y:
        #         scores[cnt, cnt2] = rmsle(np.exp(y[test_index]) - 1, y_comb)
        #     else:
        #         scores[cnt, cnt2] = rmsle(y[test_index], y_comb)
        #     th = th + .001
        #     cnt2 = cnt2 + 1
        if is_log_y:
            scores[cnt] = rmsle(np.exp(y[test_index]) - 1, y_comb)
        else:
            scores[cnt] = rmsle(y[test_index], y_comb)
        cnt = cnt + 1
    
    # for i in range(num_weight):
    #     print(scores[:, i].mean(), ' - ', scores[:, i].std(), ' weight: ', i)
    
    return scores


def predict_with_categorical(model, x, bin_size, y_train_mean):
    m, n = x.shape
    y_ = np.zeros((m, 1), dtype=np.int32)
    for i in range(m):
        curr = x[i, :]
        for j in range(n):
            if curr[j] not in model[j]:
                y_[i] = y_train_mean
                continue
            d = model[j][curr[j]]
            # select the most frequent label as label
            # selected_label = max(d, key=lambda key: d[key])
            # get weighted average of results (weights are already normalized)
            selected_label = sum([x * d[x] for x in d])
            y_[i] = y_[i] + selected_label
        # find the average value
        y_[i] = y_[i] / n
    return y_ * bin_size


def get_bayes_model(x, y, bin_size: int):
    """ counts the occurence of each label for each feature and for each value """

    m, n = x.shape
    # for each feature, keep a dictionary for frequency of values
    # model is an array of dictionary of dictionary which simply indexes feature index, feature value and label frequency
    model = [{}] * n

    for i in range(m):
        label = round(y[i] / bin_size)
        for j in range(n):
            val = x[i][j]
            if val in model[j]:
                if label in model[j][val]:
                    model[j][val][label] = model[j][val][label] + 1
                else:
                    model[j][val][label] = 1
            else:
                model[j][val] = {label: 1}

    # normalize counts of labels to [0,1]
    for feature in model:
        for val in feature:
            s = 0
            for freq in feature[val]:
                s = s + feature[val][freq]
            for freq in feature[val]:
                feature[val][freq] = feature[val][freq] / s

    return model


def run_model2(x_train, y_train, x_test):
    bin_size = 1000
    m = get_bayes_model(x_train, y_train, bin_size)
    y_train_mean = np.mean(y_train)
    return predict_with_categorical(m, x_test, bin_size, y_train_mean)


svr = make_pipeline(RobustScaler(),
                    SVR(gamma='auto'))


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

is_log_y, is_2_numeric, is_del_outlier, is_2_categorical, is_fill_missing_numeric, is_fill_missing = True, True, True, True, True, False
X, _, y = transform_features(is_log_transform_y=is_log_y, is_delete_outlier_from_train=is_del_outlier, is_2_numeric=is_2_numeric, 
                             is_2_categorical=is_2_categorical, is_fill_missing_numeric=is_fill_missing_numeric, is_fill_missing=is_fill_missing, is_fix_skewed=False, is_drop_too_uniq=False)
if not is_2_numeric:
    X = X.select_dtypes(exclude=['object'])
is_log_y2 = False
x2, _, y2 = transform_features(
    is_2_numeric=False, is_log_transform_y=is_log_y2)
x2 = x2.select_dtypes(['object'])

start_time = time.time()
score = cross_val_rmsle(svr, X, y, is_log_y)
print('SVR rmsle: ', score.mean(), ' std: ', score.std())
print(time.time() - start_time, ' passed ')
