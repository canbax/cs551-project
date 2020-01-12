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
from sklearn.preprocessing import normalize
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
        features.update(features.select_dtypes(
            numeric_dtypes).fillna(features.mean()))
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


def cross_val_rmsle(model, X, y, is_logged_y=True, x2=[], y2=[], is_logged_y2=False, weight=0):
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
        y_comb = y_ * (1-weight) + weight * y_2

        if is_log_y:
            scores[cnt] = rmsle(np.exp(y[test_index]) - 1, y_comb)
        else:
            scores[cnt] = rmsle(y[test_index], y_comb)
        cnt = cnt + 1

    return scores


def cross_val_rmsle2(model, X, y, is_logged_y=True, model2=None, x2=[], y2=[], is_logged_y2=False, weight=0):
    X = np.array(X)
    y = np.array(y)

    if len(x2) > 0:
        x2 = np.array(x2)
        y2 = np.array(y2)

    X = RobustScaler().fit_transform(X)
    num_split = 10
    kf = KFold(n_splits=num_split, shuffle=True)

    scores = np.ones((num_split, ))
    cnt = 0
    for train_index, test_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        model2.fit(x2[train_index], y2[train_index])

        y_ = model.predict(X[test_index])
        y_2 = model2.predict(x2[test_index])

        if is_log_y:
            y_ = np.exp(y_) - 1
        if is_log_y2:
            y_2 = np.exp(y_2) - 1

        y_comb = y_ * (1-weight) + weight * y_2

        if is_log_y:
            scores[cnt] = rmsle(np.exp(y[test_index]) - 1, y_comb)
        else:
            scores[cnt] = rmsle(y[test_index], y_comb)
        cnt = cnt + 1

    return scores


def predict_with_categorical(model, x, bin_size, y_train_mean, is_select_max=False, dynamic_bins=False, bins=None, corrs=None):
    m, n = x.shape
    y_ = np.zeros((m, 1), dtype=np.int32)
    for i in range(m):
        curr = x[i, :]
        feature_effects = np.zeros(n)
        for j in range(n):
            if curr[j] not in model[j]:
                # y_[i] = y_train_mean
                continue
            d = model[j][curr[j]]
            selected_label = 0
            # select the most frequent label as label
            if is_select_max:
                selected_label = max(d, key=lambda key: d[key])
                if dynamic_bins:
                    selected_label = bin_num2_val(
                        bins, max(d, key=lambda key: d[key]))
            # get weighted average of results (weights are already normalized)
            else:
                selected_label = sum([x * d[x] for x in d])
                if dynamic_bins:
                    selected_label = sum(
                        [bin_num2_val(bins, x) * d[x] for x in d])
            feature_effects = selected_label

        # find the average value
        # y_[i] = y_[i] / n
        y_[i] = np.dot(feature_effects, corrs)
    # if we use dynmaic bins values are already converted in previous steps
    if dynamic_bins:
        return y_
    return y_ * bin_size


def get_bayes_model(x, y, bin_size: int, dynamic_bins=False):
    """ counts the occurence of each label for each feature and for each value """

    m, n = x.shape
    # for each feature, keep a dictionary for frequency of values
    # model is an array of dictionary of dictionary which simply indexes feature index, feature value and label frequency
    model = [{}] * n

    # if dynamic_bins: bin_size means bin_count
    _, bins = pd.qcut(np.ravel(pd.DataFrame(y)), bin_size,
                      retbins=True, duplicates='drop')

    for i in range(m):
        label = round(y[i] / bin_size)
        if dynamic_bins:
            label = find_bin(bins, y[i])
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

    corrs = np.zeros(n)
    for i in range(n):
        corrs[i] = np.correlate(x[:, i], y)

    corrs = normalize(corrs)

    return model, bins, corrs


def find_bin(bins: np.array, val):

    bin_cnt = bins.shape[0]

    low_bin = 0
    curr_bin = int(bin_cnt / 2)
    up_bin = bin_cnt - 1

    # binary search
    while True:
        if bins[curr_bin] <= val and bins[curr_bin + 1] >= val:
            break
        if bins[curr_bin] >= val:
            up_bin = curr_bin
        if bins[curr_bin + 1] < val:
            low_bin = curr_bin
        curr_bin = int((low_bin + up_bin) / 2)

    return curr_bin


def bin_num2_val(bins, bin_idx):
    return (bins[bin_idx] + bins[bin_idx + 1]) / 2


def run_model2(x_train, y_train, x_test):
    bin_size = 1000
    dynamic_bins = True
    m, bins, corrs = get_bayes_model(x_train, y_train, bin_size, dynamic_bins)
    y_train_mean = np.mean(y_train)
    return predict_with_categorical(m, x_test, bin_size, y_train_mean, True, dynamic_bins, bins, corrs)


def find_best_param(X, y, is_log_y):
    for i in range(100):
        e = (i + 1) / 10000
        score = cross_val_rmsle(SVR(epsilon=0.029, gamma=e), X, y, is_log_y)
        print('SVR rmsle: ', score.mean(), ' std: ', score.std(), ' c: ', e)


def find_best_weight(model, X, y, is_logged_y=True, model2=None, x2=[], y2=[], is_logged_y2=False):
    w = 0.1
    for _ in range(10):
        score = cross_val_rmsle2(
            SVR(kernel='rbf', gamma='auto', epsilon=0.029), X, y, is_log_y, model2, x2, y2, False, w)
        print('SVR rmsle: ', score.mean(), ' std: ', score.std(),
              ' executed in: ', time.time() - start_time, ' w: ', w)
        w = w + 0.1


def get_submission_file(X, model, x, y, X2, x2, y2):
    X = np.array(X)

    x = np.array(x)
    rs = RobustScaler().fit(x)
    x = rs.transform(x)
    X = rs.transform(X)
    y = np.array(y)

    model.fit(x, y)
    y_ = model.predict(X)
    
    y_2 = run_model2(x2, y2, X2.select_dtypes(['object']))
    w = 0.1
    y_ = y_2 * w + y_ * (1 - w)
    y_ = np.exp(y_) - 1
    df = pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': y_})
    df.to_csv('submit.csv', index=None)


svr = SVR(kernel='rbf', gamma='auto', epsilon=0.029)

print('TEST score on CV')

is_log_y, is_2_numeric, is_del_outlier, is_2_categorical, is_fill_missing_numeric, is_fill_missing, is_fix_skew, is_drop_too_uniq \
    = True, False, True, True, True, False, True, False
X, X_test, y = transform_features(is_log_transform_y=is_log_y, is_delete_outlier_from_train=is_del_outlier, is_2_numeric=is_2_numeric,
                                  is_2_categorical=is_2_categorical, is_fill_missing_numeric=is_fill_missing_numeric, is_fill_missing=is_fill_missing,
                                  is_fix_skewed=is_fix_skew, is_drop_too_uniq=is_drop_too_uniq)

if not is_2_numeric:
    X = X.select_dtypes(exclude=['object'])

is_log_y2 = True
x2, X_test2, y2 = transform_features(
    is_2_numeric=False, is_log_transform_y=is_log_y2)
x2 = pd.get_dummies(x2.select_dtypes(['object'])).reset_index(drop=True)

start_time = time.time()
X_test2 = X_test2.select_dtypes(['object'])
# find_best_param(X, y, is_log_y)
# score = cross_val_rmsle(svr, X, y, is_log_y, x2, y2, False)
# print('SVR rmsle: ', score.mean(), ' std: ', score.std(),
#       ' executed in: ', time.time() - start_time)
# find_best_weight(SVR(kernel='linear', gamma='auto', epsilon=0.029),
#                  X, y, is_log_y, gbr, x2, y2, is_log_y2)
if not is_2_numeric:
    X_test = X_test.select_dtypes(exclude=['object'])
    X = X.select_dtypes(exclude=['object'])
get_submission_file(X_test, svr, X, y, X_test2, x2, y2)
