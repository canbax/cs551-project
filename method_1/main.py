import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


def divide_data():
    """ return numerical and categorical data respectively """
    df = pd.read_csv('data/train.csv')

    y = df['SalePrice']
    df.drop(['SalePrice'], axis=1, inplace=True)
    # train1 is numerical data types
    x1 = df.select_dtypes(exclude=['object'])
    # train2 is categorical data types
    x2 = df.select_dtypes(['object'])

    return x1, x2, y


def get_bayes_model(x, y, bin_size: int):
    """ 1 dimensional bayesian """
    