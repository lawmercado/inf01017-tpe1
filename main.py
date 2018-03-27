#! /usr/bin/python

import csv
from data.transformation import NumericDataTransformer
from ml.supervised.algorithms import knn


rows = list(csv.reader(open('diabetes.csv', 'r')))

data_transformer = NumericDataTransformer(rows, 'Outcome')

instances = data_transformer.as_instances()

folds = data_transformer.stratify(7)

for idx, fold in enumerate(folds):
    aux_folds = list(folds)  # Copy the folds
    test_instances = [instance[0] for instance in aux_folds.pop(idx)]

    train_instances = []
    for aux_fold in aux_folds:
        train_instances += aux_fold

    classified = knn(train_instances, test_instances, 5)

    print(classified)

    # Compare classified instances with the training set
    # Then generate the statistics

