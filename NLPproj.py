
# import packages
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# import pyspark
# import findspark
# import os
# from pyspark import SparkContext

# spark_location = '/spark'       # Set your own
# java8_location = '/usr/lib/jvm/java-8-openjdk-amd64'  # Set your own
# os.environ['JAVA_HOME'] = java8_location
# findspark.init(spark_home=spark_location)
# sc = SparkContext()

# define parameters
filename = "Eluvio_DS_Challenge.csv"
chunksize = 100000
dictsize = 1000
maxlen = 30
embedsize = 5

fulldf = pd.read_csv(filename)#, nrows=chunksize)
# create tokenizer object (convert words to integers)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=dictsize)
tokenizer.fit_on_texts(fulldf["title"])

# build empty RNN model
rnnmodel = keras.Sequential()
# add embedding layer (convert integers to vectors)
rnnmodel.add( keras.layers.Embedding(input_dim=dictsize, output_dim=embedsize, input_length=maxlen) )
# add LSTM layer outputs at every time
rnnmodel.add( keras.layers.LSTM(units=10, return_sequences=True, activation="sigmoid") )
# add LSTM layer outputs at every time
rnnmodel.add( keras.layers.LSTM(units=10, return_sequences=True, activation="sigmoid") )
# add LSTM layer outputs at every time
rnnmodel.add( keras.layers.LSTM(units=5, return_sequences=False, activation="sigmoid") )
# add fully connected layer to return 0/1
rnnmodel.add( keras.layers.Dense(units=5, activation="relu") )
# add fully connected layer to return value
rnnmodel.add( keras.layers.Dense(units=1) )
# compile RNN model
rnnmodel.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=1e-3))
# check RNN model summary
rnnmodel.summary()

# read in dataframe in chunks
i = 0
for df in pd.read_csv(filename, chunksize=chunksize):
    print("reading chunk", i)
    i = i + 1

    # pre-process data (df_x, df_y)
    # tokenize words to integers by pre-defined tokenizer
    df_token = tokenizer.texts_to_sequences(df["title"])
    # convert to same length to be used as x in tensorflow
    df_x = keras.preprocessing.sequence.pad_sequences(df_token, maxlen=maxlen, padding="pre", truncating="pre")
    # use log of upvotes as y value
    df_uv = np.add(df["up_votes"], 1)
    df_y = np.log(df_uv)

    # split data into train and test (train_x, train_y, test_x, test_y)
    ind = np.random.rand(len(df)) < 0.8
    train_x = df_x[ind]
    test_x = df_x[~ind]
    train_y = np.asarray(df_y[ind])
    test_y = np.asarray(df_y[~ind])
    # train / update RNN model in current chunk
    rnnmodel.fit(train_x, train_y, batch_size=100, epochs=5, validation_split=0.05)

# evaluate RNN model
result = rnnmodel.evaluate(test_x, test_y)
print("MSE: ", result)

mean = sum(df_y)/len(df_y)
var = sum(np.square(np.add(df_y, - mean)))/len(df_y)
print("compare:", var)
# predict RNN model
# result = rnnmodel.predict(pred_x)


