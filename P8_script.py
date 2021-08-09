import os
import boto3

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import udf, broadcast,countDistinct
from pyspark.mllib.clustering import KMeans


import sagemaker
from sagemaker import get_execution_role
import sagemaker_pyspark

from PIL import Image, ImageOps, ImageFilter
import cv2 as cv


from pyspark.sql.functions import col, pandas_udf, PandasUDFType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

t = time.time() 
# get region
region = boto3.session.Session().region_name
# get bucket: hang's project 8 of openclassrooms
bucket_name = 'hangp8'
role = get_execution_role()

# for exploration, we use train sample, which contains 3 categories, 1243 images
dir_prefix = 'train_sample/'

# Configure Spark to use the SageMaker Spark dependency jars
jars = sagemaker_pyspark.classpath_jars()

classpath = ":".join(jars)

# Sets the Spark master URL to connect to, run locally with all cores available: 'local[*]'
spark = (
    SparkSession.builder.config("spark.driver.extraClassPath", classpath)
    .master("local[*]")
    .getOrCreate()
)

# save img path and labels
def get_path_label_list(prefix=dir_prefix, bucket_name=bucket_name):
    s3 = boto3.resource('s3', region_name=region)
    bucket=bucket_name
    my_bucket = s3.Bucket(bucket)
    path_label_list = []
    for (bucket_name, key) in map(lambda x: (x.bucket_name, x.key), my_bucket.objects.filter(Prefix=prefix)):
        # save img path
        img_location = "s3://{}/{}".format(bucket_name, key)
        # save img label
        img_label = img_location.split('/')[-2]
        path_label_list.append((img_location, img_label))
    # remove the root folder
    return path_label_list

# create spark dataframe
def create_df(prefix=dir_prefix, bucket=bucket_name):
    data = get_path_label_list(prefix, bucket)
    columns = ['path', 'label']
    df_data = spark.createDataFrame(data).toDF(*columns)
    return df_data

def preprocess(path_img):
    
    '''for each image: create preprocessed image array and orb descriptors'''
    
    my_bucket = bucket_name
    # get in-bucket path
    key = os.path.relpath(path_img, 's3://'+bucket_name+'/')
    
    image_object = boto3.resource("s3", region_name=region).Bucket(my_bucket).Object(key)
    img = Image.open(image_object.get()["Body"])
    
    # preprocess
    # remove 1% extreme lightest and darkest pixels then maximize image contrast
    tmp1 = ImageOps.autocontrast(img, cutoff=1)
    # equalize image histogram to creat a uniform distribution of grayscale
    tmp2 = ImageOps.equalize(tmp1)
    img_out = tmp2
    img_prep = np.array(img_out).flatten().tolist()
    
    # descriptors
    # set max feature to retain = 50
    orb = cv.ORB_create(nfeatures=50)
    _, des = orb.detectAndCompute(np.array(img_out),None)
    img_des = ([] if des is None else des.flatten().tolist())
    
    return (img_prep, img_des)

# create a initial pyspark dataframe, add path and label of image
df_data = create_df()

# add preprocessing to df_data
schema = StructType([
    StructField("img_prep", ArrayType(IntegerType()), False),
    StructField("img_orb_des", ArrayType(IntegerType()), False)])

preprocess_udf = udf(preprocess, schema)

df_data = df_data.withColumn("Output", preprocess_udf('path'))

# ORB Descriptor processing
# convert flatten descriptors to orb ndarray descriptors (n, 32)
rdd_data = df_data.rdd.map(lambda x: Row(path = x.path,\
                                         label = x.label,\
                                         img_prep = np.array(x['Output']['img_prep']).reshape(100, 100, 3),\
                                         img_orb_des = np.array(x['Output']['img_orb_des']).reshape(-1, 32)
                                        )
                          )

# k is n_cluster, suppose 10*categories
def create_model(rdd, k):
    """Create kmeans model to create visual words cluster"""
    all_desp = rdd.flatMap(lambda row: [i for i in row['img_orb_des']])
    model = KMeans.train(all_desp, k, maxIterations=10, initializationMode="random", seed=42)
    return model


def bovw(row, model, k):
    n_des = len(row.img_orb_des)
    bovw = np.zeros(k)
    for des_ in row.img_orb_des:
        cluster = model.predict(des_)
        #normalize freq
        bovw[cluster] += 1/n_des
    return Row(path=row.path, label=row.label, bovw=bovw.tolist())

# define number of visual words
n_cat = df_data.select(countDistinct('label')).take(1)[0][0]
k = 10 * n_cat
model_kmeans = create_model(rdd_data, k)
bovw_rdd = rdd_data.map(lambda x: bovw(x, model_kmeans, k))
df_final = bovw_rdd.toDF()
df_final.show(10)
print('total loading and preprocess time: {},\n{} images,\n{} categories'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-t)), df_data.count(), n_cat))