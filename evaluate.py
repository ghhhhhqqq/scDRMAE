"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from __future__ import print_function
import time
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

def evaluate(label, pred):
    # nmi = metrics.normalized_mutual_info_score(label, pred)
    # ari = metrics.adjusted_rand_score(label, pred)
    # # f = metrics.fowlkes_mallows_score(label, pred)
    # # pred_adjusted = get_y_preds(
    # #     label, pred, max(len(set(label)), len(set(pred))))
    # ami = metrics.adjusted_mutual_info_score(label, pred)
    # acc = metrics.accuracy_score(pred, label)
    acc = accuracy_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    ami = adjusted_mutual_info_score(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    return nmi, ari, acc,ami


def evaluate_others(label, pred):
    ami = metrics.adjusted_mutual_info_score(label, pred)
    homo, comp, v_mea = metrics.homogeneity_completeness_v_measure(label, pred)
    return ami, homo, comp, v_mea


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred
