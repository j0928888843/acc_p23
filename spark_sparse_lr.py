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


def parse_line_key(line):

    # gen a uniqe key for each sample
    key = hash(line + str(random.randint(1, 4294967295)))
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
    value = (label, (np.array(feature_ids), np.array(feature_vals)))
    return (key, value)


def init_sample(t):
    key, value = t
    features = value[1]
    feature_ids = features[0]
    feature_vals = features[1]
    length = len(feature_vals)

    weight_init_value = 0.001
    init_feature_vals = np.ones(length) * weight_init_value
    return (key, (feature_ids, init_feature_vals))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# compute logarithm of a number but thresholding the number to avoid logarithm of 0


def safe_log(x):
    if x < 1e-15:
        x = 1e-15
    return math.log(x)

# compute the gradient descent updates and cross-entropy loss for an RDD partition


def gd_partition(samples):
    local_updates = defaultdict(float)
    local_weights_array = weights_array_bc.value
    cross_entropy_loss = 0

    # compute and accumulate updates for each data sample in the partition
    for sample in samples:
        label = sample[0]
        features = sample[1]
        feature_ids = features[0]
        feature_vals = features[1]
        # fetch the relevant weights for this sample as a numpy array
        local_weights = np.take(local_weights_array, feature_ids)
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
    accumulated_updates = sps.csr_matrix(
        (local_updates.values(),
         local_updates.keys(),
         [0, len(local_updates)]),
        shape=(1, num_features))
    return [(cross_entropy_loss, accumulated_updates)]


def gd_partition_key(samples):
    local_updates = defaultdict(float)
    local_weights_array = weights_array_bc.value
    cross_entropy_loss = 0

    # compute and accumulate updates for each data sample in the partition
    for sample in samples:
        (key, value) = sample
        label = value[0]
        features = value[1]
        feature_ids = features[0]
        feature_vals = features[1]
        # fetch the relevant weights for this sample as a numpy array
        local_weights = np.take(local_weights_array, feature_ids)
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
    accumulated_updates = sps.csr_matrix(
        (local_updates.values(),
         local_updates.keys(),
         [0, len(local_updates)]),
        shape=(1, num_features))
    return [(cross_entropy_loss, accumulated_updates)]


def gd_partition_key_test(samples):
    local_updates = defaultdict(float)
    # local_weights_array = weights_array_bc.value
    cross_entropy_loss = 0

    # compute and accumulate updates for each data sample in the partition
    for sample in samples:
        key, payload = sample
        value, update = payload
        label, features = value

        feature_ids = features[0]
        feature_vals = features[1]
        update_feature_ids = update[0]
        update_weight_vals = update[1]

        # fetch the relevant weights for this sample as a numpy array

        #local_weights = np.take(local_weights_array, feature_ids)
        local_weights = update_weight_vals

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
    return [(cross_entropy_loss, result_update_list)]


def tuple_to_loss_only(t):
    cross_entropy_loss = t[0]
    return cross_entropy_loss


def tuple_to_list_only(t):
    update_list_t = t[1]
    return update_list_t


def identical(x):
    return x


def my_add(a, b):
    return a + b


def sample_to_weight_sample_list_pair(t):
    (k, v) = t

    # key = hash(sample string)
    # value = (label, (np.array(feature_ids), np.array(feature_vals)))

    label, features = v
    feature_ids = features[0]
    # veature_vals = features[1]

    result_list = []
    for feature_id in feature_ids:
        result_list.append((feature_id, [k]))
    return result_list


# def list_add(l1, l2):
#     # return l1.extend(l2)
#     s1 = set(l1)
#     s2 = set(l2)
#     s_sum = s1.union(s2)
#     return list(s_sum)


def list_add(l1, l2):

    return l1 + l2

def list_add2(l1, l2):

    return l1 + l2

def to_sample_update_pair(t):
    key, value = t  # key is feature_id, value is (update_value, [hash(sample1),....])
    feature_id = key
    update_value = value[0]
    hash_sample_list = value[1]
    results = []  # each element is ( hash(sample), feature_id:update_value )
    for hash_sample in hash_sample_list:
        a = feature_id
        # b = np.array(float(update_value))
        b = update_value
        results.append((hash_sample, (a, b)))

    return results


def add_features(ta, tb):
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    feature_ids_sum = np.append(feature_ids_a, feature_ids_b)
    update_values_sum = np.append(update_values_a, update_values_b)
    return (feature_ids_sum, update_values_sum)


def add_features2(ta, tb):
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    feature_ids_sum = np.append(feature_ids_a, feature_ids_b)
    update_values_sum = np.append(update_values_a, update_values_b)
    return (feature_ids_sum, update_values_sum)

def test(t):
    k, v = t
    main = v[0]
    update = v[1]
    return update


def add_array_element(ta, tb):
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    update_values_sum = update_values_a + update_values_b

    result = (feature_ids_a, update_values_sum)
    return result


def merge_features(value):
    ta, tb = value
    feature_ids_a, update_values_a = ta
    feature_ids_b, update_values_b = tb
    # indexs = np.argsort(feature_ids_b)
    indexs = feature_ids_b.argsort()
    sorted_update_values_b = update_values_b[indexs[::1]]
    update_values_sum = update_values_a + sorted_update_values_b
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


def to_pid_samples_pair_tt(samples):
    # below 3 approach gets same results
    # 1.
    #
    # l = []
    # for sample in samples:
    #     l.append(sample)
    # return [l]
    #
    # 2.
    # yield list(samples)
    #
    # 3.
    return [list(samples)]


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

        # fetch the relevant weights for this sample as a numpy array

        #local_weights = np.take(local_weights_array, feature_ids)
        # local_weights = update_weight_vals

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
    # pid_samples_rdd (the input data)
    # , preservesPartitioning=True) \
    # #.partitionBy(num_partitions)
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

        # first_cur_pid_fids_weight_rdd = cur_pid_fids_weight_rdd.take(1)
        # first_cur_pid_samples_fids_weights_rdd = cur_pid_samples_fids_weights_rdd.take(1)

        # decay step size to ensure convergence
        step_size *= 0.95
        print "iteration: %d, cross-entropy loss: %f" % (0, my_loss)

        # loss_updates_rdd = cur_pid_samples_fids_weights_rdd.map(gd_partition_new, preservesPartitioning=True)
        # my_loss2 = loss_updates_rdd.map(tuple_to_loss_only).treeReduce(lambda x, y: x + y, 7)

        # print "pid_samples_rdd.take(1)", pid_samples_rdd.take(1)
        # print "pid_samples_rdd.count()", pid_samples_rdd.count()
        # print "fid_pids_rdd.take(1) inverted index pair", fid_pids_rdd.take(1)
        # print "pid_fids_rdd_take(1)", pid_fids_rdd.take(1)
        # print "cur_pid_fids_fvals_rdd", cur_pid_fids_fvals_rdd.take(1)
        # print "cur_fid_weight_rdd.take(4)", cur_fid_weight_rdd.take(4)
        # print "cur_pid_samples_fids_fvals_rdd.take(1)", cur_pid_samples_fids_fvals_rdd.take(1)[0][1][1]
        print"my_loss", my_loss
        # print "loss_updates_rdd.count()", loss_updates_rdd.count()
        print "======================================================="
    print"my_loss", loss_list

    with open(loss_file, "w") as loss_fobj:
        for loss in loss_list:
            loss_fobj.write(str(loss) + "\n")

    sc.stop()

    '''
    #===============================================

    # the RDD that contains parsed data samples, which are reused during
    # training
    samples_rdd = text_rdd.map(parse_line_key, preservesPartitioning=True) \
        .partitionBy(num_partitions).persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK)
    # force samples_rdd to be created
    num_samples = samples_rdd.count()
    # initialize weights as a local array
    # weights_array = np.ones(num_features) * weight_init_value

    invert_index_rdd = samples_rdd.flatMap(sample_to_weight_sample_list_pair).reduceByKey(list_add) \
        .partitionBy(num_partitions).persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK)

    # force invert_index_rdd to be created
    num_invert_index = invert_index_rdd.count()

    my_loss_list = []
    loss_list = []
    cur_rdd = samples_rdd.map(init_sample, preservesPartitioning=True).partitionBy(num_partitions)
    cur_samples_rdd = samples_rdd.join(cur_rdd, num_partitions)
    # cur_samples_rdd_test = cur_samples_rdd.map(test).collect()
    # sample_part_num = samples_rdd.getNumPartitions()
    # part_num = cur_samples_rdd.getNumPartitions()
    #
    # cur_rdd = init_update_rdd.map(lambda x: x)

    # force cur_samples_rdd to be created
    num_cur_samples_rdd = cur_samples_rdd.count()

    # force cur_rdd to be created
    num_cur_rdd = cur_rdd.count()

    for iteration in range(0, num_iterations):
        # broadcast weights array to workers
        # weights_array_bc = sc.broadcast(weights_array)
        # compute gradient descent updates in parallel

        # loss_updates_rdd = samples_rdd.mapPartitions(gd_partition_key)

        # collect and sum up the and updates cross-entropy loss over all
        # partitions

        # ret = loss_updates_rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        # loss = ret[0]
        # updates = ret[1]
        # loss_list.append(loss)
        # weights_array += updates.toarray().squeeze()

        # loss_updates_rdd_test = samples_rdd.mapPartitions(gd_partition_key_test_old)
        loss_updates_rdd_test = cur_samples_rdd.mapPartitions(gd_partition_key_test, preservesPartitioning=True)
        my_loss = loss_updates_rdd_test.map(tuple_to_loss_only).treeReduce(lambda x, y: x + y, 7)
        my_loss_list.append(my_loss)
        tmp_0_rdd = loss_updates_rdd_test.map(tuple_to_list_only, preservesPartitioning=True) \
            .flatMap(lambda x: x) \
            .reduceByKey(lambda x, y: x + y, numPartitions=num_partitions)  # \
        #    .partitionBy(num_partitions)
        weight_update_rdd = tmp_0_rdd  # .partitionBy(num_partitions)
        result_rdd = weight_update_rdd.join(invert_index_rdd, num_partitions)

        tmp_1_rdd = result_rdd.flatMap(to_sample_update_pair)
        samples_update_rdd = tmp_1_rdd.reduceByKey(add_features, numPartitions=num_partitions)

        #cur_rdd = cur_rdd.union(samples_update_rdd).reduceByKey(add_array_element).partitionBy(num_partitions)
        tmp_2_rdd = cur_rdd.join(samples_update_rdd, num_partitions).mapValues(merge_features)  # .partitionBy(num_partitions)
        cur_rdd = tmp_2_rdd  # .partitionBy(num_partitions)
        cur_samples_rdd = samples_rdd.join(cur_rdd, num_partitions)

        # force cur_rdd and cur_samples_rdd to be created
        num_cur_rdd = cur_rdd.count()
        num_cur_samples_rdd = cur_samples_rdd.count()
        # first_cur_rdd = cur_rdd.take(1)
        # first_cur_samples_rdd = cur_samples_rdd.take(1)

        # decay step size to ensure convergence
        step_size *= 0.95
        print "iteration: %d, cross-entropy loss: %f" % (iteration, my_loss)
        # the lsat line of for loop is destroy()
        # weights_array_bc.destroy()
    loss_list = my_loss_list
    print '=============================================='
    # print 'loss_list', loss_list
    print 'my_loss_list', my_loss_list
    print '=============================================='
    # print 'cur_samples_rdd_1', ans
    # print 'cur_samples_rdd_2', ans2
    # print 'part_num', part_num
    # print 'sample_part_num', sample_part_num
    print '=============================================='
    # write the cross-entropy loss to a local file
    with open(loss_file, "w") as loss_fobj:
        for loss in loss_list:
            loss_fobj.write(str(loss) + "\n")

    sc.stop()

    '''
