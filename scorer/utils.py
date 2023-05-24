import pdb
import logging
import argparse
import os

import sys

MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50]

def _compute_average_precision(gold_labels, pred_labels):
    """ Computes Average Precision. """

    precisions = []
    num_correct = 0
    num_positive = sum([1 if v == 1 else 0 for k, v in gold_labels.items()])

    for i, line_number in enumerate(pred_labels):
        if gold_labels[line_number] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec

def _compute_average_precision(gold_labels, ranked_lines):
    """ Computes Average Precision. """

    precisions = []
    num_correct = 0
    num_positive = sum([1 if v == 1 else 0 for k, v in gold_labels.items()])

    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec


def _compute_reciprocal_rank(gold_labels, ranked_lines):
    """ Computes Reciprocal Rank. """
    rr = 0.0
    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            rr += 1.0 / (i + 1)
            break
    return rr


def _compute_precisions(gold_labels, ranked_lines, threshold):
    """ Computes Precision at each line_number in the ordered list. """
    precisions = [0.0] * threshold
    threshold = min(threshold, len(ranked_lines))

    for i, line_number in enumerate(ranked_lines[:threshold]):
        if gold_labels[line_number] == 1:
            precisions[i] += 1.0

    for i in range(1, threshold): # accumulate
        precisions[i] += precisions[i - 1]
    for i in range(1, threshold): # normalize
        precisions[i] /= i+1
    return precisions


def get_threshold_line_format(thresholds, last_entry_name):
    threshold_line_format = '{:<30}' + "".join(['@{:<9}'.format(ind) for ind in thresholds])
    if last_entry_name:
        threshold_line_format = threshold_line_format + '{:<9}'.format(last_entry_name)
    return threshold_line_format

def print_thresholded_metric(title, thresholds, data, last_entry_name=None, last_entry_value=None):
    line_separator = '=' * 120
    threshold_line_format = get_threshold_line_format(thresholds, last_entry_name)
    items = data
    if last_entry_value is not None:
        items = items + [last_entry_value]
    logging.info(threshold_line_format.format(title))
    logging.info('{:<30}'.format("") + "".join(['{0:<10.4f}'.format(item) for item in items]))
    logging.info(line_separator)

def print_single_metric(title, value):
    line_separator = '=' * 120
    logging.info('{:<30}'.format(title) + '{0:<10.4f}'.format(value))
    logging.info(line_separator)

def print_metrics_info(line_separator):
    logging.info('Description of the evaluation metrics: ')
    logging.info('!!! THE OFFICIAL METRIC USED FOR THE COMPETITION RANKING IS MEAN AVERAGE PRECISION (MAP) !!!')
    logging.info('R-Precision is Precision at R, where R is the number of relevant line_numbers for the evaluated set.')
    logging.info('Average Precision is the precision@N, estimated only @ each relevant line_number and then averaged over the number of relevant line_numbers.')
    logging.info('Reciprocal Rank is the reciprocal of the rank of the first relevant line_number in the list of predictions sorted by score (descendingly).')
    logging.info('Precision@N is precision estimated for the first N line_numbers in the provided ranked list.')
    logging.info('The MEAN versions of each metric are provided to average over multiple debates (each with separate prediction file).')
    logging.info(line_separator)
    logging.info(line_separator)
