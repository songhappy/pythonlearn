export ANALYTICS_ZOO_HOME=/Users/guoqiong/intelWork/git/analytics-zoo/dist/
export SPARK_HOME=/Users/guoqiong/intelWork/tools/spark/spark-2.4.3-bin-hadoop2.7
MASTER=local[8]
bash ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 8  \
    --driver-memory 2g  \
    --total-executor-cores 8  \
    --executor-cores 8  \
    --executor-memory 2g \
