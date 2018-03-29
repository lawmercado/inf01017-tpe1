#! /usr/bin/python

from __future__ import division
from __future__ import print_function
import csv
from data.transformation import NumericDataTransformer
from ml.supervised.validation import knn_kcrossvalidation
from ml.supervised.validation import knn_repeatedkcrossvalidation

rows = list(csv.reader(open('diabetes.csv', 'r')))

data_transformer = NumericDataTransformer(rows, 'Outcome')

instances = data_transformer.as_instances()

print("Cross validation with 3 folds and 3 nearest neighbors:")
print(knn_kcrossvalidation(data_transformer, 3, 3))

#print("Cross validation with 5 folds and 5 nearest neighbors:")
#print(knn_kcrossvalidation(data_transformer, 5, 5))
#print("Cross validation with 10 folds and 5 nearest neighbors:")
#print(knn_kcrossvalidation(data_transformer, 5, 10))
#print("Repeated cross validation with 10 iterations, 10 folds and 3 nearest neighbours:")
#print(knn_repeatedkcrossvalidation(data_transformer, 3, 10, 10))
#print("Repeated cross validation with 10 iterations, 10 folds and 5 nearest neighbours:")
#print(knn_repeatedkcrossvalidation(data_transformer, 5, 10, 10))
#print("Repeated cross validation with 10 iterations, 10 folds and 7 nearest neighbours:")
#print(knn_repeatedkcrossvalidation(data_transformer, 7, 10, 10))
