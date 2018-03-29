#! /usr/bin/python

from __future__ import division
from __future__ import print_function
import csv
from data.transformation import NumericDataTransformer
from ml.supervised.evaluation import knn_kcrossvalidation
from ml.supervised.evaluation import knn_repeatedkcrossvalidation
from ml.supervised.evaluation import get_statistics

rows = list(csv.reader(open('diabetes.csv', 'r')))

data_transformer = NumericDataTransformer(rows, 'Outcome')

print("\nCross validation with 5 folds and 5 nearest neighbors:")
print(get_statistics(knn_kcrossvalidation(data_transformer, 5, 5)))

print("\nCross validation with 10 folds and 5 nearest neighbors:")
print(get_statistics(knn_kcrossvalidation(data_transformer, 5, 10)))

print("\nRepeated cross validation with 10 iterations, 10 folds and 3 nearest neighbours:")
print(get_statistics(knn_repeatedkcrossvalidation(data_transformer, 3, 10, 10)))

print("\nRepeated cross validation with 10 iterations, 10 folds and 5 nearest neighbours:")
print(get_statistics(knn_repeatedkcrossvalidation(data_transformer, 5, 10, 10)))

print("\nRepeated cross validation with 10 iterations, 10 folds and 7 nearest neighbours:")
print(get_statistics(knn_repeatedkcrossvalidation(data_transformer, 7, 10, 10)))
