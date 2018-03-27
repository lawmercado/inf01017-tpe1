#! /usr/bin/python

import csv


def numeric_file_import(csv_file):

    rows = list(csv.reader(csv_file))

    attrs = rows.pop(0)

    data = {}

    for idx, attr in enumerate(attrs):
        data[attr] = []

        for row in rows:
            try:
                data[attr].append(float(row[idx].strip()))
            except ValueError:
                pass

    return data
