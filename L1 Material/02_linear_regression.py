import pandas as pd
import numpy as np

data_house = pd.read_csv('house_price.tsv', sep='\t')

x = data_house.as_matrix(['size'])
y = data_house.as_matrix(['price'])

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

x = data_house.as_matrix(['size', 'Taxes'])
y = data_house.as_matrix(['price'])

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

sst = sum((y - np.mean(y)) ** 2)
ssr = sum((regr.predict(x) - np.mean(y)) ** 2)
sse = sum((regr.predict(x) - y) ** 2)

print('Total sum of squares:', sst)
print('Explained sum of squares:', ssr)
print('Residual sum of squares:', sse)
print('R^2 score computed from score function:', regr.score(x, y))
print('R^2 score computed from ssr / sst:', ssr / sst)

np.random.seed(2017)
'''
from datetime import datetime
np.random.seed(np.random.seed(datetime.now().microsecond))
'''

train = np.random.choice([True, False], len(x), replace=True, p=[0.9,0.1])
x_train = x[train,:]
y_train = y[train]
x_test = x[~train,:]
y_test = y[~train]
regr.fit(x_train, y_train)
print('R^2 score: %.2f' % regr.score(x_test, y_test))

from sklearn import metrics

y_pred = regr.predict(x_test)
metrics.explained_variance_score(y_test, y_pred)
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)

from sklearn import preprocessing

poly2 = preprocessing.PolynomialFeatures(2)
poly3 = preprocessing.PolynomialFeatures(3)

x2 = poly2.fit_transform(x)
x3 = poly3.fit_transform(x)

x_train = x2[train,:]
x_test = x2[~train,:]
regr.fit(x_train, y_train)
print('R^2 score: %.2f' % regr.score(x_test, y_test))

x_train = x3[train,:]
x_test = x3[~train,:]
regr.fit(x_train, y_train)
print('R^2 score: %.2f' % regr.score(x_test, y_test))

regr_no_intercept = linear_model.LinearRegression(fit_intercept=False)

x_train = x2[train,:]
x_test = x2[~train,:]
regr.fit(x_train, y_train)
regr_no_intercept.fit(x_train, y_train)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

print('Coefficients:', regr_no_intercept.coef_)
print('Intercept:', regr_no_intercept.intercept_)

import math

def map_to_higher_dim(orig_data, terms):
    mapped = []
    for x in orig_data:
        x_higher = []
        for d in terms:
            v = 1.0
            for pos, exponent in d.items():
                v *= math.pow(x[pos], exponent)
            x_higher.append(v)
        mapped.append(x_higher)
    return np.asarray(mapped)

terms = [{0:2}, {1:2}, {0:1,1:1}]
x_mapped = map_to_higher_dim(x, terms)
x_train = x_mapped[train,:]
x_test = x_mapped[~train,:]
regr.fit(x_train, y_train)
print('R^2 score: %.2f' % regr.score(x_test, y_test))

from sklearn import datasets
diabetes = datasets.load_diabetes()
x = diabetes.data[:,:4]
y = diabetes.target
regr = linear_model.LinearRegression()
regr.fit(x, y)
sgd = linear_model.SGDRegressor(n_iter=100000, penalty='none')
sgd.fit(x, y)
regr.score(x, y)
sgd.score(x, y)
print(regr.coef_, regr.intercept_)
print(sgd.coef_, sgd.intercept_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = diabetes.data[:,[2,8]]
y = diabetes.target
regr = linear_model.LinearRegression()
regr.fit(x, y)
steps = 40
lx0 = np.arange(min(x[:,0]), max(x[:,0]), (max(x[:,0]) - min(x[:,0])) / steps).reshape(steps,1)
lx1 = np.arange(min(x[:,1]), max(x[:,1]), (max(x[:,1]) - min(x[:,1])) / steps).reshape(steps,1)
xx0, xx1 = np.meshgrid(lx0, lx1)
xx = np.zeros(shape = (steps,steps,2))
xx[:,:,0] = xx0
xx[:,:,1] = xx1
x_stack = xx.reshape(steps ** 2, 2)
y_stack = regr.predict(x_stack)
yy = y_stack.reshape(steps, steps)

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(x[:,0], x[:,1], y, color = 'red')
ax.plot_surface(xx0, xx1, yy, rstride=1, cstride=1)
plt.show()
