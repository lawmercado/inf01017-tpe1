#! /usr/bin/python

from __future__ import division
import csv
from data.transformation import NumericDataTransformer
from ml.supervised.validation import knn_kcrossvalidation
from ml.supervised.validation import knn_repeatedkcrossvalidation

rows = list(csv.reader(open('diabetes.csv', 'r')))

data_transformer = NumericDataTransformer(rows, 'Outcome')

instances = data_transformer.as_instances()

print(knn_kcrossvalidation(data_transformer, 5, 5))
print(knn_kcrossvalidation(data_transformer, 5, 10))
print(knn_repeatedkcrossvalidation(data_transformer, 3, 10, 10))
print(knn_repeatedkcrossvalidation(data_transformer, 5, 10, 10))
print(knn_repeatedkcrossvalidation(data_transformer, 7, 10, 10))
