import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import seaborn as sns; print("Seaborn", sns.__version__)
import matplotlib.pyplot as plt;

import sys
import findspark
import pandas as pd

import pyarrow as pa
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

from pyspark.sql import Window
import sys

from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.linalg import SparseVector, DenseVector
from io import BytesIO

from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import *

from pyspark.ml.linalg import Vectors, VectorUDT
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.fs_utils import FilesystemResolver
import pyarrow as pa

def sparse_encode(v):
    v = DenseVector(v)
    arr = list([int(x) for x in v])
    memfile = BytesIO()
    np.save(memfile, np.array(arr).astype(np.uint8))
    return bytearray(memfile.getvalue())

def embedding_encode(l):
    arr = list([float(item) for sublist in l for item in sublist])
    memfile = BytesIO()    
    np.save(memfile, np.array(arr).astype(np.float32))
    return bytearray(memfile.getvalue())

spark = SparkSession\
            .builder\
            .config("spark.executor.memory", "64g")\
            .config("spark.driver.memory", "256g")\
            .config("spark.class", "sortByKeyDf")\
            .config("spark.executor.memoryOverhead", "512m")\
            .config("spark.driver.maxResultSize", "128g")\
            .config("spark.applicationId", "embeddings")\
            .config("spark.sql.broadcastTimeout",  "62000")\
            .config("spark.local.dir", "/var/tmp")\
            .appName("test")\
            .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.enabled", "false")

print(spark.sparkContext.version)

LIMIT = 100000000
MAX_EVENTS = 100
SERV_DAY_IN = "0, 1"
serv_day_list = [0, 1]

lu_bill = spark.read.parquet('./data/phd/lu_bill.parquet').registerTempTable("lu_bill")
readmit = spark.read.parquet('./data/phd/readmissions.parquet').registerTempTable("readmit")

print('loaded lookup tables')

import torch
chkpnt = torch.load('./code/poincare-embeddings-master/trained_embeddings/10-batch.pth', map_location='cuda:0')
embeddings = chkpnt['embeddings']
objects = np.array(chkpnt['objects']).astype('int64')
np_embeddings = embeddings.cpu().numpy().astype('float32')
objects = np.append(objects, '-1')
np_embeddings = np.append(np_embeddings, [[0] * 10], axis=0)

i = 7
j = 4

print(str(i), str(j))

quarter_logic = "phd_201" + str(i) + str(j)
quarter_logic_WILD  = quarter_logic.replace('*', 'WILD')
print(quarter_logic, quarter_logic_WILD)


pat_noapr = spark.read.format("csv")\
           .option("inferSchema", "true")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .load(["hdfs://" + quarter_logic + "_pat_noapr.txt"])\
           .registerTempTable("pat")

pat_bill = spark.read.format("csv")\
           .option("inferSchema", "true")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .load(["hdfs://" + quarter_logic + "_patbill.txt"])\
           .registerTempTable('bill')

readmit = spark.sql('select * from readmit')

windowSpec = Window.partitionBy("MEDREC_KEY").orderBy("DISC_MON")
readmit.withColumn('readmit_time', f.lead(readmit.DAYS_FROM_PRIOR, 1).over(windowSpec)).registerTempTable("readmittimetable")

cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ")
pats = spark.sql(query).select(cols).cache()
spark.sql(query).select(cols).registerTempTable('pats')

query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
         "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
         "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
bill = spark.sql(query)

cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ") 
print(query)

pats = spark.sql(query).select(cols)
pats.registerTempTable("pat_subset")

query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
         "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat_subset) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
         "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
print(query)
bill = spark.sql(query)

query = ('select pats.*, IFNULL(readmittimetable.readmit_time,9999) as readmit_time FROM pats ' +
         'LEFT JOIN readmittimetable on (pats.pat_key = readmittimetable.PAT_KEY)')
print(query)
merged_pats = spark.sql(query).cache()

assembled = bill.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"])\
                                                .orderBy(['STD_CHG_CODE', f.desc('SERV_DAY')])))

blo_df = assembled.select('PAT_KEY').distinct()
n_to_array = f.udf(lambda n : [n] * MAX_EVENTS, ArrayType(IntegerType()))
blo_df = blo_df.withColumn('PAT_KEY', f.explode(n_to_array(blo_df.PAT_KEY)))
blo_df = blo_df.withColumn('expandID', f.monotonically_increasing_id())
blo_df = blo_df.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"]).orderBy(['expandID'])))

joined = assembled.join(blo_df, ['PAT_KEY', 'RANK'], 'right').orderBy(['PAT_KEY', 'rank'], ascending=False)
joined = joined.dropDuplicates(['PAT_KEY', 'RANK', 'STD_CHG_CODE'])

joined = joined.fillna(-1, subset=['STD_CHG_CODE', 'SERV_DAY', 'STD_QTY'])
joined.printSchema()

object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                             ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])
embedded = joined.join(object_spark, "STD_CHG_CODE", 'left')


null_value = f.array([f.lit(0.0)] * MAX_EVENTS)

# If you want a different type you can change it like this
null_value = null_value.cast('array<float>')

# Keep the value when there is one, replace it when it's null
embedded = (embedded.withColumn('EMBEDDING', f.when(embedded['EMBEDDING'].isNull(), null_value)\
            .otherwise(embedded['EMBEDDING'])))

model = PipelineModel.load("./onehot")
transformed = model.transform(merged_pats)

object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                             ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])

spark_df = embedded.join(transformed, ['PAT_KEY'], 'left')
spark_df = spark_df.withColumn('mortalityLabel', f.when(merged_pats['disc_status']==20, 1).otherwise(0))
spark_df = spark_df.withColumn('losLabel', f.when(merged_pats['LOS']>6, 1).otherwise(0))
spark_df = spark_df.withColumn('costLabel', merged_pats['pat_cost'])
spark_df = spark_df.withColumn('readmitLabel', f.when(merged_pats['readmit_time']<31, 1).otherwise(0))

emb_dim = MAX_EVENTS*10
compressed_df = (spark_df\
                 .orderBy(['PAT_KEY', 'rank'], ascending=False)\
                 .groupby('PAT_KEY')\
                 .agg(f.collect_list("EMBEDDING").alias('EMBEDDING'),
                      f.first('FEATURES').alias('FEATURES'),
                      f.first('mortalityLabel').alias('mortalityLabel'),
                      f.first('losLabel').alias('losLabel'),
                      f.first('costLabel').alias('costLabel'),
                      f.first('readmitLabel').alias('readmitLabel')
                 )
                )

sparse_encode_udf = f.udf(sparse_encode, BinaryType())
embedding_encode_udf = f.udf(embedding_encode, BinaryType())

out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

frameSchema = Unischema('frameSchema', [
    UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
    UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
    UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
    UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
])

out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

frameSchema = Unischema('frameSchema', [
    UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
    UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
    UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
    UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
])

import pyspark.sql.functions as f

out_df = out_df.withColumn('randid', f.monotonically_increasing_id())
rout_df = out_df.sort('randid').registerTempTable("rout_df")

rowgroup_size_mb = 256
FSR = FilesystemResolver('./dashboard/', hdfs_driver='libhdfs')
filesystem_factory = FSR._filesystem_factory

for k in range(1,10):
    query = ('select PAT_KEY, EMBEDDING, FEATURES, mortalityLabel, losLabel, costLabel, readmitLabel from rout_df WHERE randid % 10 == ' + str(k))
    print(query)
    fold = spark.sql(query)
    output_url= ('./dashboard/' + 'Features_' + quarter_logic +  '_random_fold'+ str(k) +'.parquet')
    print(output_url)
    with materialize_dataset(spark, output_url, frameSchema, rowgroup_size_mb, filesystem_factory=filesystem_factory):
        fold\
            .write \
            .mode('overwrite') \
            .parquet(output_url)


for i in [5]:
    for j in [2,3,4]:
        print(str(i), str(j))
        
        quarter_logic = "phd_201" + str(i) + str(j)
        quarter_logic_WILD  = quarter_logic.replace('*', 'WILD')
        print(quarter_logic, quarter_logic_WILD)
        pat_noapr = spark.read.format("csv")\
                   .option("inferSchema", "true")\
                   .option("delimiter", "|")\
                   .option("header", "true")\
                   .load(["./" + quarter_logic + "_pat_noapr.txt"])\
                   .registerTempTable("pat")

        pat_bill = spark.read.format("csv")\
                   .option("inferSchema", "true")\
                   .option("delimiter", "|")\
                   .option("header", "true")\
                   .load(["./" + quarter_logic + "_patbill.txt"])\
                   .registerTempTable('bill')
        
        readmit = spark.sql('select * from readmit')
        windowSpec = Window.partitionBy("MEDREC_KEY").orderBy("DISC_MON")
        readmit.withColumn('readmit_time', f.lead(readmit.DAYS_FROM_PRIOR, 1).over(windowSpec)).registerTempTable("readmittimetable")

        cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
        'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
        'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

        query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ")
        pats = spark.sql(query).select(cols).cache()
        spark.sql(query).select(cols).registerTempTable('pats')

        query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
                 "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
                 "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
        bill = spark.sql(query)

        cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
        'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
        'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

        query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ") 
        print(query)

        #PAT_KEY, disc_status
        pats = spark.sql(query).select(cols)
        pats.registerTempTable("pat_subset")

        query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
                 "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat_subset) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
                 "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
        print(query)
        bill = spark.sql(query)

        query = ('select pats.*, IFNULL(readmittimetable.readmit_time,9999) as readmit_time FROM pats ' +
                 'LEFT JOIN readmittimetable on (pats.pat_key = readmittimetable.PAT_KEY)')
        print(query)
        merged_pats = spark.sql(query).cache()

        assembled = bill.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"])\
                                                        .orderBy(['STD_CHG_CODE', f.desc('SERV_DAY')])))

        blo_df = assembled.select('PAT_KEY').distinct()
        n_to_array = f.udf(lambda n : [n] * MAX_EVENTS, ArrayType(IntegerType()))
        blo_df = blo_df.withColumn('PAT_KEY', f.explode(n_to_array(blo_df.PAT_KEY)))
        blo_df = blo_df.withColumn('expandID', f.monotonically_increasing_id())
        blo_df = blo_df.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"]).orderBy(['expandID'])))

        joined = assembled.join(blo_df, ['PAT_KEY', 'RANK'], 'right').orderBy(['PAT_KEY', 'rank'], ascending=False)
        joined = joined.dropDuplicates(['PAT_KEY', 'RANK', 'STD_CHG_CODE'])

        joined = joined.fillna(-1, subset=['STD_CHG_CODE', 'SERV_DAY', 'STD_QTY'])
        joined.printSchema()

        object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                                     ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])
        embedded = joined.join(object_spark, "STD_CHG_CODE", 'left')


        null_value = f.array([f.lit(0.0)] * MAX_EVENTS)

        # If you want a different type you can change it like this
        null_value = null_value.cast('array<float>')

        # Keep the value when there is one, replace it when it's null
        embedded = (embedded.withColumn('EMBEDDING', f.when(embedded['EMBEDDING'].isNull(), null_value)\
                    .otherwise(embedded['EMBEDDING'])))

        model = PipelineModel.load("./onehot")
        transformed = model.transform(merged_pats)

        object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                                     ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])

        spark_df = embedded.join(transformed, ['PAT_KEY'], 'left')
        spark_df = spark_df.withColumn('mortalityLabel', f.when(merged_pats['disc_status']==20, 1).otherwise(0))
        spark_df = spark_df.withColumn('losLabel', f.when(merged_pats['LOS']>6, 1).otherwise(0))
        spark_df = spark_df.withColumn('costLabel', merged_pats['pat_cost'])
        spark_df = spark_df.withColumn('readmitLabel', f.when(merged_pats['readmit_time']<31, 1).otherwise(0))

        emb_dim = MAX_EVENTS*10
        compressed_df = (spark_df\
                         .orderBy(['PAT_KEY', 'rank'], ascending=False)\
                         .groupby('PAT_KEY')\
                         .agg(f.collect_list("EMBEDDING").alias('EMBEDDING'),
                              f.first('FEATURES').alias('FEATURES'),
                              f.first('mortalityLabel').alias('mortalityLabel'),
                              f.first('losLabel').alias('losLabel'),
                              f.first('costLabel').alias('costLabel'),
                              f.first('readmitLabel').alias('readmitLabel')
                         )
                        )

        sparse_encode_udf = f.udf(sparse_encode, BinaryType())
        embedding_encode_udf = f.udf(embedding_encode, BinaryType())

        out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
        #         out_df = compressed_df
        out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

        frameSchema = Unischema('frameSchema', [
            UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
            UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
            UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
            UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
            UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
        ])
        out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
        #         out_df = compressed_df
        out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

        frameSchema = Unischema('frameSchema', [
            UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
            UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
            UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
            UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
            UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
        ])
        out_df = out_df.withColumn('randid', f.monotonically_increasing_id())
        rout_df = out_df.sort('randid').registerTempTable("rout_df")

        rowgroup_size_mb = 256
        FSR = FilesystemResolver('hdfs://dashboard/', hdfs_driver='libhdfs')
        filesystem_factory = FSR._filesystem_factory
        for k in range(10):
            query = ('select PAT_KEY, EMBEDDING, FEATURES, mortalityLabel, losLabel, costLabel, readmitLabel from rout_df WHERE randid % 10 == ' + str(k))
            print(query)
            fold = spark.sql(query)
            output_url= ('./Features_' + quarter_logic +  '_random_fold'+ str(k) +'.parquet')
            print(output_url)
            with materialize_dataset(spark, output_url, frameSchema, rowgroup_size_mb, filesystem_factory=filesystem_factory):
                fold\
                    .write \
                    .mode('overwrite') \
                    .parquet(output_url)
i = 3
j = 4
print(str(i), str(j))

quarter_logic = "phd_201" + str(i) + str(j)
quarter_logic_WILD  = quarter_logic.replace('*', 'WILD')
print(quarter_logic, quarter_logic_WILD)
pat_noapr = spark.read.format("csv")\
           .option("inferSchema", "true")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .load(["hdfs://" + quarter_logic + "_pat_noapr.txt"])\
           .registerTempTable("pat")

pat_bill = spark.read.format("csv")\
           .option("inferSchema", "true")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .load(["hdfs://" + quarter_logic + "_patbill.txt"])\
           .registerTempTable('bill')

readmit = spark.sql('select * from readmit')
windowSpec = Window.partitionBy("MEDREC_KEY").orderBy("DISC_MON")
readmit.withColumn('readmit_time', f.lead(readmit.DAYS_FROM_PRIOR, 1).over(windowSpec)).registerTempTable("readmittimetable")

cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ")
pats = spark.sql(query).select(cols).cache()
spark.sql(query).select(cols).registerTempTable('pats')

query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
         "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
         "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
bill = spark.sql(query)

cols = ['pat_key', 'ADM_MON', 'DISC_MON_SEQ', 'ADM_SOURCE', 'POINT_OF_ORIGIN', 'ADM_TYPE', 'MART_STATUS',
'AGE', 'GENDER', 'RACE', 'HISPANIC_IND', 'STD_PAYOR', 'PROV_ID', 'ADMPHY_SPEC', 'ADM_PHY', 'ATTPHY_SPEC', 
'ATT_PHY', 'PROJ_WGT', 'disc_status', 'pat_cost', 'los']

query = ("SELECT * FROM pat WHERE LOS > 1 LIMIT " + str(LIMIT) + " ") 
print(query)

#PAT_KEY, disc_status
pats = spark.sql(query).select(cols)
pats.registerTempTable("pat_subset")

query = ("SELECT PAT_KEY, STD_CHG_CODE, SERV_DAY, STD_QTY FROM bill b " +
         "WHERE b.PAT_KEY IN (SELECT PAT_KEY FROM pat_subset) AND b.SERV_DAY IN (" + SERV_DAY_IN + ") " +
         "ORDER BY PAT_KEY ASC, SERV_DAY ASC") 
print(query)
bill = spark.sql(query)

query = ('select pats.*, IFNULL(readmittimetable.readmit_time,9999) as readmit_time FROM pats ' +
         'LEFT JOIN readmittimetable on (pats.pat_key = readmittimetable.PAT_KEY)')
print(query)
merged_pats = spark.sql(query).cache()

assembled = bill.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"])\
                                                .orderBy(['STD_CHG_CODE', f.desc('SERV_DAY')])))

blo_df = assembled.select('PAT_KEY').distinct()
n_to_array = f.udf(lambda n : [n] * MAX_EVENTS, ArrayType(IntegerType()))
blo_df = blo_df.withColumn('PAT_KEY', f.explode(n_to_array(blo_df.PAT_KEY)))
blo_df = blo_df.withColumn('expandID', f.monotonically_increasing_id())
blo_df = blo_df.withColumn('RANK', f.dense_rank().over(Window.partitionBy(["PAT_KEY"]).orderBy(['expandID'])))

joined = assembled.join(blo_df, ['PAT_KEY', 'RANK'], 'right').orderBy(['PAT_KEY', 'rank'], ascending=False)
joined = joined.dropDuplicates(['PAT_KEY', 'RANK', 'STD_CHG_CODE'])

joined = joined.fillna(-1, subset=['STD_CHG_CODE', 'SERV_DAY', 'STD_QTY'])
joined.printSchema()

object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                             ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])
embedded = joined.join(object_spark, "STD_CHG_CODE", 'left')


null_value = f.array([f.lit(0.0)] * MAX_EVENTS)
null_value = null_value.cast('array<float>')

# Keep the value when there is one, replace it when it's null
embedded = (embedded.withColumn('EMBEDDING', f.when(embedded['EMBEDDING'].isNull(), null_value)\
            .otherwise(embedded['EMBEDDING'])))

model = PipelineModel.load("./onehot")
transformed = model.transform(merged_pats)

object_spark = spark.createDataFrame(zip(objects.tolist(), range(objects.shape[0]), np_embeddings.tolist()), 
                             ['STD_CHG_CODE', 'POSTION', 'EMBEDDING'])

spark_df = embedded.join(transformed, ['PAT_KEY'], 'left')
spark_df = spark_df.withColumn('mortalityLabel', f.when(merged_pats['disc_status']==20, 1).otherwise(0))
spark_df = spark_df.withColumn('losLabel', f.when(merged_pats['LOS']>6, 1).otherwise(0))
spark_df = spark_df.withColumn('costLabel', merged_pats['pat_cost'])
spark_df = spark_df.withColumn('readmitLabel', f.when(merged_pats['readmit_time']<31, 1).otherwise(0))

emb_dim = MAX_EVENTS*10
compressed_df = (spark_df\
                 .orderBy(['PAT_KEY', 'rank'], ascending=False)\
                 .groupby('PAT_KEY')\
                 .agg(f.collect_list("EMBEDDING").alias('EMBEDDING'),
                      f.first('FEATURES').alias('FEATURES'),
                      f.first('mortalityLabel').alias('mortalityLabel'),
                      f.first('losLabel').alias('losLabel'),
                      f.first('costLabel').alias('costLabel'),
                      f.first('readmitLabel').alias('readmitLabel')
                 )
                )

sparse_encode_udf = f.udf(sparse_encode, BinaryType())
embedding_encode_udf = f.udf(embedding_encode, BinaryType())

out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
#         out_df = compressed_df
out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

frameSchema = Unischema('frameSchema', [
    UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
    UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
    UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
    UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
])
out_df = compressed_df.withColumn('FEATURES', sparse_encode_udf('FEATURES'))
#         out_df = compressed_df
out_df = out_df.withColumn('EMBEDDING', embedding_encode_udf('EMBEDDING'))

frameSchema = Unischema('frameSchema', [
    UnischemaField('PAT_KEY', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('EMBEDDING', np.float32, (emb_dim,), NdarrayCodec(), False),
    UnischemaField('FEATURES', np.uint8, (1320,), NdarrayCodec(), False),
    UnischemaField('mortalityLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('losLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('costLabel', np.float64, (), ScalarCodec(FloatType()), False),
    UnischemaField('readmitLabel', np.uint8, (), ScalarCodec(IntegerType()), False),
])
out_df = out_df.withColumn('randid', f.monotonically_increasing_id())
rout_df = out_df.sort('randid').registerTempTable("rout_df")

rowgroup_size_mb = 256
FSR = FilesystemResolver('hdfs://dashboard/', hdfs_driver='libhdfs')
filesystem_factory = FSR._filesystem_factory
k = 9
query = ('select PAT_KEY, EMBEDDING, FEATURES, mortalityLabel, losLabel, costLabel, readmitLabel from rout_df WHERE randid % 10 == ' + str(k))
print(query)
fold = spark.sql(query)
output_url= ('hdfs://dashboard/' + 'Features_' + quarter_logic +  '_random_fold'+ str(k) +'.parquet')
print(output_url)
with materialize_dataset(spark, output_url, frameSchema, rowgroup_size_mb, filesystem_factory=filesystem_factory):
    fold\
        .write \
        .mode('append') \
        .parquet(output_url)

print('done')
