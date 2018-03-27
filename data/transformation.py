#! /usr/bin/python

import itertools


def __get_average(data):
    averages = {key: 0 for key in data}

    for key in data:
        for item in data[key]:
            averages[key] += item

        averages[key] = averages[key] / len(data[key])

    return averages


def __get_std_deviations(data, averages):
    std_deviations = {key: 0 for key in data}

    for key in data:
        for item in data[key]:
            std_deviations[key] += (item - averages[key])**2

        std_deviations[key] = (std_deviations[key] / (len(data[key]) - 1))**0.5

    return std_deviations


def normalize(data):
    """
    Normalizes the data represented by an dictionary

    :param list data: {<attribute1>: [<values1>], ...}
    :return: A list with the normalized data
    :rtype: list
    """

    averages = __get_average(data)
    std_deviations = __get_std_deviations(data, averages)

    normalized_data = { key: [] for key in data }

    for key in data:
        for item in data[key]:
            normalized_item = (item - averages[key]) / std_deviations[key]
            normalized_data[key].append(normalized_item)

    return normalized_data


def to_instances(data, class_attr):
    classes = data[class_attr]
    data.pop(class_attr, None)

    normalized_data = normalize(data)

    instances = []

    for _ in itertools.repeat(None, len(classes)):
        instances.append(())

    for key in normalized_data:
        for idx, value in enumerate(normalized_data[key]):
            instances[idx] = instances[idx] + (value,)

    instances = [(instance, classes[idx]) for idx, instance in enumerate(instances)]

    return instances
