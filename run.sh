#!/bin/sh

# The spark_sparse_lr.py program takes in the following command line arguments, which are all
# required:
# 1) data_path: the HDFS path where the training data set is stored
# 2) num_features: the total number of feature dimensions of the training data
# 3) num_iterations: the number of training iterations
# 4) step_size: the gradient descent step size
# 5) loss_file: a local path to store the cross-entropy loss for each iteration
# 
# # For example, run this program with: (/kdd10)
# <path-to-spark-submit> spark_sparse_lr.py /kdda 20216830 10 2e-6 loss_kdda
PROJ_PATH=`dirname $0`
# echo "hello"
# echo $PROJ_PATH
# echo $1
#  --master spark://ip-172-31-11-159.ec2.internal:7077 \
# --total-executor-cores 20
# --conf spark.default.parallelism=63
/root/spark/bin/spark-submit --driver-memory 5g --executor-memory 5g \
 --conf spark.driver.cores=4 \
 --conf spark.driver.maxResultSize=10g \
$PROJ_PATH/spark_sparse_lr.py $1 $2 $3 $4 $5
