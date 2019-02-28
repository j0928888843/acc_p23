# By Jinliang Wei (jinlianw@cs.cmu.edu)
# Copyright (c) 2017 Carnegie Mellon University
# For use in 15-719 only
# All other rights reserved.

# This program trains a logistic regression model using the gradient descent
# algorithm.
# The program takes in the following command line arguments, which are all
# required:
# 1) data_path: the HDFS path where the training data set is stored
# 2) num_features: the total number of feature dimensions of the training data
# 3) num_iterations: the number of training iterations
# 4) step_size: the gradient descent step size
# 5) loss_file: a local path to store the cross-entropy loss for each iteration
#
# For example, run this program with:
# <path-to-spark-submit> spark_sparse_lr.py /kdda 20216830 10 2e-6 loss_kdda

import pyspark
import sys
import numpy as np
import math
from datetime import datetime
import scipy.sparse as sps
import string
from collections import defaultdict
import random
# import helper as h

# parse a line of the training data file to produce a data sample record


def parse_line(line):
    parts = line.split()
    label = int(parts[0])
    # the program requires binary labels in {0,1}
    # the dataset may have binary labels -1 and 1, we convert all -1 to 0
    label = 0 if label == -1 else label
    feature_ids = []
    feature_vals = []
    for part in parts[1:]:
        feature = part.split(":")
        # the datasets have feature ids in [1, N] we convert them
        # to [0, N - 1] for array indexing
        feature_ids.append(int(feature[0]) - 1)
        feature_vals.append(float(feature[1]))
    return (label, (np.array(feature_ids), np.array(feature_vals)))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# compute logarithm of a number but thresholding the number to avoid logarithm of 0


def safe_log(x):
    if x < 1e-15:
        x = 1e-15
    return math.log(x)

def tuple_to_loss_only(t):
    cross_entropy_loss = t[0]
    return cross_entropy_loss


def tuple_to_list_only(t):
    update_list_t = t[1]
    return update_list_t


def list_add2(l1, l2):

    return l1 + l2


def add_features2(ta, tb):
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    feature_ids_sum = np.append(feature_ids_a, feature_ids_b)
    update_values_sum = np.append(update_values_a, update_values_b)
    return (feature_ids_sum, update_values_sum)

def add_array_element(ta, tb):
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    update_values_sum = update_values_a + update_values_b

    result = (feature_ids_a, update_values_sum)
    return result


def merge_features2(value):
    ta, tb = value
    feature_ids_a, update_weights_a = ta  # feature_ids is np.int64, update_weights is np.float64
    feature_ids_b, update_weights_b = tb
    mydict = {}
    length_b = len(feature_ids_b)
    for i in range(0, length_b):
        mydict[feature_ids_b[i]] = [update_weights_b[i]]

    # merge the weight with same feature id
    length_a = len(feature_ids_a)
    for i in range(0, length_a):
        update_weights_a[i] = update_weights_a[i] + mydict[feature_ids_a[i]]

    result = (feature_ids_a, update_weights_a)
    return result


def to_pid_samples_pair(pid, samples):
    return [(pid, list(samples))]

def to_fid_pid_list(t):  # generate  (feature_id,[pid]) and (pid, fid_array) for invert index
    pid, samples = t
    result_list = []
    feature_id_set = set()
    for sample in samples:
        label = sample[0]
        features = sample[1]
        feature_ids = features[0]
        feature_vals = features[1]

        # only add (fid, [pid]) to result_list when fid has not been added before
        for feature_id in feature_ids:
            if feature_id not in feature_id_set:
                result_list.append((feature_id, [pid]))
                feature_id_set.add(feature_id)

    feature_ids_array = np.array(list(feature_id_set))
    return (result_list, (pid, feature_ids_array))


def init_feature_value(t):
    pid, feature_ids = t
    length = len(feature_ids)
    weight_init_value = 0.001
    init_feature_vals = np.ones(length) * weight_init_value
    return (pid, (feature_ids, init_feature_vals))


def init_cur_fid_weight(t):
    # init cur_fid_weight from fid_pids_rdd
    fid = t[0]
    # weight = np.array([float(0.001)])
    weight = np.float64(0.001)
    return (fid, weight)


def gd_partition_new(t):
    # convert cur_pid_samples_fids_fvals_rdd to loss and weight by GD
    local_updates = defaultdict(float)

    cross_entropy_loss = 0

    pid, payload = t
    samples, fids_fvals = payload
    partition_feature_ids = fids_fvals[0]  # np.array [np.int64]
    partition_weight_vals = fids_fvals[1]  # np.array [np.float64]

    weight_dict = {}
    length = len(partition_feature_ids)
    for i in range(0, length):
        weight_dict[partition_feature_ids[i]] = partition_weight_vals[i]

    # compute and accumulate updates for each data sample in the partition
    for sample in samples:
        label = sample[0]
        features = sample[1]
        feature_ids = features[0]  # np.array [np.int64]
        feature_vals = features[1]  # np.array [np.float64]

        temp_list = []
        for idx, j in enumerate(feature_ids):
            temp_list.append(weight_dict[j])

        local_weights = np.array(temp_list)

        # given the current weights, the probability of this sample belonging to
        # class '1'
        pred = sigmoid(feature_vals.dot(local_weights))
        diff = label - pred
        # the L2-regularlized gradients
        gradient = diff * feature_vals - reg_param * local_weights
        sample_update = step_size * gradient

        for i in range(0, feature_ids.size):
            local_updates[feature_ids[i]] += sample_update[i]

        # compute the cross-entropy loss, which is an indirect measure of the
        # objective function that the gradient descent algorithm optimizes for
        if label == 1:
            cross_entropy_loss -= safe_log(pred)
        else:
            cross_entropy_loss -= safe_log(1 - pred)

    result_update_list = []
    for feature_id, local_update in local_updates.iteritems():
        result_update_list.append((feature_id, local_update))
    return (cross_entropy_loss, result_update_list)


def to_pid_fid_update_weight(t):
    # t is fid_weight_pids_rdd
    # ouput should be (pid, (fid,update_weight))
    fid, payload = t  # fid is feature_id, payload is (update_weight, pids)
    feature_id = fid
    update_weight = payload[0]
    pid_list = payload[1]
    results = []  # each element is ( pid, (feature_id,update_weight) )
    for pid in pid_list:
        a = feature_id  # type np.int64
        # b = np.array(float(update_weight))
        b = update_weight  # type np.float64
        results.append((pid, (a, b)))

    return results


if __name__ == "__main__":
    data_path = sys.argv[1]
    num_features = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    step_size = float(sys.argv[4])
    loss_file = sys.argv[5]

    # for all test cases, your weights should all be initialized to this value
    weight_init_value = 0.001
    # the step size is multiplicatively decreased each iteration at this rate
    step_size_decay = 0.95
    # the L2 regularization parameter
    reg_param = 0.01

    # total number of cores of your Spark slaves
    # num_cores = 4
    num_cores = 64 * 32
    # for simplicity, the number of partitions is hardcoded
    # the number of partitions should be configured based on data size
    # and number of cores in your cluster
    # num_partitions = num_cores * 4
    num_partitions = num_cores * 1
    conf = pyspark.SparkConf().setAppName("SparseLogisticRegressionGD")
    sc = pyspark.SparkContext(conf=conf)

    text_rdd = sc.textFile(data_path, minPartitions=num_partitions)
    pid_samples_rdd = text_rdd.map(parse_line)  \
        .mapPartitionsWithIndex(to_pid_samples_pair) \
        .persist(pyspark.storagelevel.StorageLevel.DISK_ONLY)

    # fid_pids_rdd (inverted_index rdd pairs for weight distrbution)
    fid_pids_rdd = pid_samples_rdd.map(to_fid_pid_list, preservesPartitioning=True) \
        .map(lambda t : t[0], preservesPartitioning=True).flatMap(lambda x: x) \
        .reduceByKey(list_add2) \
        .partitionBy(num_partitions) \
        .persist(pyspark.storagelevel.StorageLevel.DISK_ONLY)
    # pid_fids_rdd (for wight distrubution part 2)
    pid_fids_rdd = pid_samples_rdd.map(to_fid_pid_list, preservesPartitioning=True)\
        .map(lambda t : t[1], preservesPartitioning=True)  \
        .partitionBy(num_partitions).persist(pyspark.storagelevel.StorageLevel.DISK_ONLY)
    # init feature values to 0.001
    cur_pid_fids_weights_rdd = pid_fids_rdd.map(init_feature_value, preservesPartitioning=True)

    # init cur_fid_weight_rdd
    # cur_fid_weight_rdd = fid_pids_rdd.map(init_cur_fid_weight, preservesPartitioning=True) \

    # init cur_pid_samples_fids_weights
    cur_pid_samples_fids_weights_rdd = pid_samples_rdd.join(cur_pid_fids_weights_rdd, num_partitions)

    # force cur_pid_samples_fids_weights_rdd to be created
    # num_cur_pid_samples_fids_weights_rdd = cur_pid_samples_fids_weights_rdd.count()

    # force cur_pid_fids_weights_rdd to be created
    # num_cur_pid_fids_weights_rdd = cur_pid_fids_weights_rdd.count()

    loss_list = []
    for iteration in range(0, num_iterations):

        loss_updates_rdd = cur_pid_samples_fids_weights_rdd.map(gd_partition_new, preservesPartitioning=True)
        my_loss = loss_updates_rdd.map(tuple_to_loss_only).treeReduce(lambda x, y: x + y, 7)
        loss_list.append(my_loss)
        if iteration == (num_iterations - 1):  # finish last iteration after get the loss
            break
        tmp_0_rdd = loss_updates_rdd.map(tuple_to_list_only, preservesPartitioning=True) \
            .flatMap(lambda x: x) \
            .reduceByKey(lambda x, y: x + y, numPartitions=num_partitions)   \
            .partitionBy(num_partitions)
        fid_fval_rdd = tmp_0_rdd  # .partitionBy(num_partitions)
        fid_weight_pids_rdd = fid_fval_rdd.join(fid_pids_rdd, num_partitions)

        tmp_1_rdd = fid_weight_pids_rdd.flatMap(to_pid_fid_update_weight)
        pid_fids_update_weights_rdd = tmp_1_rdd.reduceByKey(add_features2, numPartitions=num_partitions)

        #cur_rdd = cur_rdd.union(cur_pid_fid_update_weight_rdd).reduceByKey(add_array_element).partitionBy(num_partitions)
        tmp_2_rdd = cur_pid_fids_weights_rdd.join(pid_fids_update_weights_rdd, num_partitions).mapValues(merge_features2)  # .partitionBy(num_partitions)
        cur_pid_fids_weights_rdd = tmp_2_rdd  # .partitionBy(num_partitions)

        cur_pid_samples_fids_weights_rdd = pid_samples_rdd.join(cur_pid_fids_weights_rdd, num_partitions)

        # force rdd to be created
        # num_cur_pid_fids_weights_rdd = cur_pid_fids_weights_rdd.count()
        # num_cur_pid_samples_fids_weights_rdd = cur_pid_samples_fids_weights_rdd.count()
        cur_pid_fids_weights_rdd.persist(pyspark.storagelevel.StorageLevel.DISK_ONLY)
        cur_pid_samples_fids_weights_rdd.persist(pyspark.storagelevel.StorageLevel.DISK_ONLY)

        # decay step size to ensure convergence
        step_size *= 0.95
        print "iteration: %d, cross-entropy loss: %f" % (0, my_loss)
        print"my_loss", my_loss
        # print "loss_updates_rdd.count()", loss_updates_rdd.count()
        print "======================================================="

    print"my_loss", loss_list

    with open(loss_file, "w") as loss_fobj:
        for loss in loss_list:
            loss_fobj.write(str(loss) + "\n")

    sc.stop()
