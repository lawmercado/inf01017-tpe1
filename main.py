#! /usr/bin/python

from data.csv_importer import numeric_file_import
from data.transformation import to_instances
from ml.algorithms import knn


data = numeric_file_import(open('diabetes.csv', 'r'))

instances = to_instances(data, 'Outcome')

# Here whe apply the k-fold cross validation

classified = knn(instances, test_instances, 5)

print(classified)
