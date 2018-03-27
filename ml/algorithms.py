#! /usr/bin/python


def __euclidean_distance(pa, pb):
    distance = 0

    for idx, position in enumerate(pa):
        distance += ((position - pb[idx])**2)

    return distance**0.5

def knn(instances, test_instances, k):
    """
    Calculates the k nearest neighbor's and predicts the class of the new test_instances (tries)

    :param list instances: A list of tuples like [((<attributes>), <classification>), ...]
    :param list test_instances: The testing instances, composed of a list of attribute tuples like [(<attributes>), ...]
    :return: A list with the classification related to the test instances
    :rtype: list
    """

    classified = []

    for test_instance in test_instances:
        distances = []

        for instance in instances:
            distance = __euclidean_distance(instance[0], test_instance)

            distances.append(distance)

        distances_idx = list(range(len(distances)))
        distances_idx.sort(key=lambda y: distances[y])

        knn_classes = [instances[i][1] for i in distances_idx[0:k]]

        counters = {key: knn_classes.count(key) for key in knn_classes}

        winner_counter = 0
        winner = 0

        for key in counters:
            if counters[key] > winner_counter:
                winner_counter = counters[key]
                winner = key

        classified.append((test_instance, winner))

    return classified
