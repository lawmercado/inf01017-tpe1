#! /usr/bin/python

import random

class NumericDataTransformer(object):

    __attr = []
    __data = []
    __class_attr = ""

    def __init__(self, raw_data, class_attr):
        self.__data = raw_data
        self.__attr = self.__data.pop(0)
        self.__class_attr = class_attr

    def by_attributes(self):
        data = {}

        for idx, attr in enumerate(self.__attr):
            data[attr] = []

            for row in self.__data:
                try:
                    data[attr].append(float(row[idx].strip()))
                except ValueError:
                    pass

        return data

    def __get_average(self):
        data = self.by_attributes()

        averages = {key: 0 for key in data}

        for key in data:
            for item in data[key]:
                averages[key] += item

            averages[key] = averages[key] / len(data[key])

        return averages

    def __get_std_deviations(self, averages):
        data = self.by_attributes()

        std_deviations = {key: 0 for key in data}

        for key in data:
            for item in data[key]:
                std_deviations[key] += (item - averages[key]) ** 2

            std_deviations[key] = (std_deviations[key] / (len(data[key]) - 1)) ** 0.5

        return std_deviations

    def normalize(self):
        """
        Normalizes the data represented by an dictionary

        :param dictionary data: {<attribute1>: [<values1>], ...}
        :return: A list with the normalized data
        :rtype: list
        """

        data = self.by_attributes()

        averages = self.__get_average()
        std_deviations = self.__get_std_deviations(averages)

        normalized_data = {key: [] for key in data}

        for key in data:
            for item in data[key]:
                normalized_item = (item - averages[key]) / std_deviations[key]
                normalized_data[key].append(normalized_item)

        return normalized_data

    def as_instances(self):
        data = self.by_attributes()

        classes = data[self.__class_attr]
        data.pop(self.__class_attr)

        normalized_data = self.normalize()

        instances = []

        for x in range(0, len(classes)):
            instances.append(())

        for key in normalized_data:
            for idx, value in enumerate(normalized_data[key]):
                instances[idx] = instances[idx] + (value,)

        instances = [(instance, classes[idx]) for idx, instance in enumerate(instances)]

        return instances

    def by_class_attribute_values(self):
        instances = self.as_instances()

        data = {instance[1]: [] for instance in instances}

        for idx, instance in enumerate(instances):
            data[instance[1]].append(idx)

        return data

    def stratify(self, k_folds):
        """
        Divide the data into k folds, maintaining the main proportion

        :param integer k_folds: Number of folds
        :return: The folds
        :rtype: list
        """

        random.seed(None)

        instances = self.as_instances()
        data = self.by_class_attribute_values()

        folds = [[] for i in range(0, k_folds)]

        instances_per_fold = round(len(instances) / k_folds)

        for yi in data:
            yi_proportion = len(data[yi]) / len(instances)

            counter = round(yi_proportion * instances_per_fold)

            while counter > 0:
                try:
                    for idx in range(0, k_folds):
                        instance = instances[data[yi].pop(random.randint(0, len(data[yi]) - 1))]

                        folds[idx].append(instance)

                    --counter

                except (ValueError, IndexError):
                    break

        return folds

