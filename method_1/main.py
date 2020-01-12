import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import warnings
import time
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from view_dict import tk_tree_view

def get_train_data():
    """ return numerical and categorical data respectively """
    df = pd.read_csv('data/train.csv')

    y = df['SalePrice']
    df.drop(['SalePrice', 'Id'], axis=1, inplace=True)
    # x1 is numerical data types
    x1 = df.select_dtypes(exclude=['object'])
    # x2 is categorical data types
    x2 = df.select_dtypes(['object'])

    x1 = x1.to_numpy()
    x2 = x2.to_numpy()
    y = y.to_numpy()

    # scale to [0,1] range
    # x1 = MinMaxScaler(feature_range=(0, 1), copy=False).fit_transform(x1)

    # mean 0 variance 1
    # x1 = StandardScaler(copy=False).fit_transform(x1)

    # x1 = RobustScaler().fit_transform(x1)

    # x1 = np.log(x1 + 1)

    return x1, x2, y


def get_test_data():
    """ return numerical and categorical data respectively """
    df = pd.read_csv('data/test.csv')

    df.drop(['Id'], axis=1, inplace=True)
    # train1 is numerical data types
    x1 = df.select_dtypes(exclude=['object'])
    # train2 is categorical data types
    x2 = df.select_dtypes(['object'])

    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(x1)

    return x1.to_numpy(), x2.to_numpy()


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


def predict_with_categorical(model, x, bin_size):
    m, n = x.shape
    y_ = np.zeros((m, 1), dtype=np.int32)
    for i in range(m):
        curr = x[i, :]
        for j in range(n):
            d = model[j][curr[j]]
            # select the most frequent label as label
            # selected_label = max(d, key=lambda key: d[key])
            # get weighted average of results (weights are already normalized)
            selected_label = sum([x * d[x] for x in d])
            y_[i] = y_[i] + selected_label
        # find the average value
        y_[i] = y_[i] / n
    return y_ * bin_size


def replace_nan2mean(x, col_mean=[]):
    if len(col_mean) == 0:
        col_mean = np.nanmean(x, axis=0)

    # find indices where nan value is present
    idxs = np.where(np.isnan(x))

    # replace inds with avg of column
    x[idxs] = np.take(col_mean, idxs[1])
    return x


def predict_with_numerical(x, y, x_test):
    x = replace_nan2mean(x)
    x_test = replace_nan2mean(x_test, np.mean(x, axis=0))
    clf = SVR(gamma='scale', kernel='linear')
    clf.fit(x, y)
    scores = cross_val_score(clf, x_test, y, cv=5)
    return clf.predict(x_test)


def predict():
    x1, x2, y = get_train_data()
    x1_test, x2_test = get_test_data()

    t = time.time()
    # y_1 = predict_with_numerical(x1, y, x1)

    bin_size = 1000
    m = get_bayes_model(x2, y, bin_size)
    
    tk_tree_view(m[0])
    
    y_2 = predict_with_categorical(m, x2, bin_size)

    print(rmsle(y, y_1))
    print(rmsle(y, y_2))
    a = 1
    print(rmsle(y, y_1 * (1 - a) + y_2 * a))

    # get_submission_file(y_1)
    print(str(time.time() - t) + ' seconds passed')
    return y_1
    # return (y_1 + y_2) / 2


def get_submission_file(y):
    df = pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': y})
    df.to_csv('submit.csv', index=None)


def rmsle(y, y_):
    """ A function to calculate Root Mean Squared Logarithmic Error (RMSLE) """
    assert len(y) == len(y_)
    return np.sqrt(np.mean((np.log(1 + y) - np.log(1 + y_))**2))


y = predict()

# print(y_)
# print(np.unique(y_))



