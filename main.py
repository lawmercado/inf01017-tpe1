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

    # Compare classified instances with the test set
    accuracy = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for (predicted_instance, test_instance) in zip(classified, test_instances):
        if predicted_instance[1] == test_instance[1]:
           accuracy += 1
           if predicted_instance[1] == 1:
               true_positive += 1
        elif predicted_instance[1] == 1:
            false_positive += 1
        elif predicted_instance[1] == 0:
            false_negative += 1

    # Then generate the statistics
    accuracy = accuracy / len(classified)  # vai dar atribuir int ou float?
    rev = true_positive / (true_positive + false_negative)
    prec = true_positive / (true_positive + false_positive)
    f_measure = 2 * (prec * rev) / (prec + rev)

    # Save statistics (list de tuplas?)