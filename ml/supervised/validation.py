#! /usr/bin/python

from __future__ import division
from ml.supervised.algorithms import knn


def knn_kcrossvalidation(data_transformer, knn_factor, k_folds):
    folds = data_transformer.stratify(k_folds)

    measures = {"acc": [], "f-measure": []}

    for idx, fold in enumerate(folds):
        aux_folds = list(folds)  # Copy the folds
        test_instances = [instance[0] for instance in aux_folds.pop(idx)]

        train_instances = []
        for aux_fold in aux_folds:
            train_instances += aux_fold

        classified = knn(train_instances, test_instances, knn_factor)

        # Compare classified instances with the test set
        correct_classifications = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for (predicted_instance, test_instance) in zip(classified, fold):
            if predicted_instance[1] == test_instance[1]:
                correct_classifications += 1
                if predicted_instance[1] == 1:
                    true_positive += 1

            elif predicted_instance[1] == 1:
                false_positive += 1

            elif predicted_instance[1] == 0:
                false_negative += 1

        # Then generate the statistics
        acc = correct_classifications / len(classified)
        measures["acc"].append(acc)

        rev = true_positive / (true_positive + false_negative)

        prec = true_positive / (true_positive + false_positive)

        f_measure = 2 * (prec * rev) / (prec + rev)
        measures["f-measure"].append(f_measure)

    return __get_statistics(measures)


def knn_repeatedkcrossvalidation(data_transformer, knn_factor, k_folds, reptitions):
    measures = {"acc": [], "f-measure": []}

    for x in range(0, reptitions):
        measure = knn_kcrossvalidation(data_transformer, knn_factor, k_folds)

        measures["acc"].append(measure["acc"][0])
        measures["f-measure"].append(measure["f-measure"][0])

    return __get_statistics(measures)


def __get_statistics(measures):
    statistics = {}

    for id in measures:
        acc = 0
        for measure in measures[id]:
            acc += measure

        avg = acc / len(measures[id])

        f_acc = 0
        for measure in measures[id]:
            f_acc += (measure - avg) ** 2

        stdd = (f_acc / (len(measures[id]) - 1)) ** 0.5

        statistics[id] = (avg, stdd)

    return statistics
