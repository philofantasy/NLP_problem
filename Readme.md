# NLP Challenge by Jiayi Fu

# Introduction

This is a NLP Code Chanllenge project on a dataset about titles and votes. The data set containing 522278 rows following 8 columns: time_created, date_created, up_votes, down_votes, title, over_18, author, category.

Based on this dataset, the problem I came up with is to predict the popularity purely based on the title. I would like to forecast the total number of up-votes when a news article is published long enough. This kind o fproblem is meaninful, beacuse for example, news platform could find potential popular articles on the front page and attract more customers. 

I used recurrent neural network (RNN) model to build the complicated regression / prediciton framework, and uses Python 3 with Tensorflow package to implement the method and finished model evaluation. To deal with huge size of data, only a chunk of data is read into memory and the model is trained gradually in iterations.

# Notes on Backgroud

By exploratory analysis, I ignored other information here for simplicity: 

Firstly, the date / time created column is ignored. Beacuse when the time passed is long enough (for example, in this data set, all articles are published at least several years ago), the total number of up-votes is amost stable and will not change much anymore. Readers are interested in newly published news instead of news of several years old. This is also justifyed by the conclusion from network science / graph theory with long term adjustment, which is a reasonable model to discribe the model of news / publishment. 

The number down-votes are also ignored. Since most of the articles have 0 downvotes, it is more efficient to only look at the up-votes. If necessary in the future, the difference or adjusted ratio (avoid dividing by 0) of up / down votes could be used. It is also possible to use scaled ratings as dependent variable instead of just up-votes.

I also ignored the over 18 and category information. Beacuse of the prevalent "False" in over_18 column and "worldnews" in category column, these two columns do not provide significant enough information to my analysis. For simplicity of the model that concentrated on NLP, these two columns are dropped.

Author information is kind of unstable over long time, although it is true that there might be some popular authors that always creates articles with great amount of up-votes. According to the data set, the authors look like users instead of media companies, which means they are more likely to be come-and-go writters over long span of time, instead of reliable news soures. It will take too much time and space to deal with the author information, espcially when many authors may suddenly jump in, only publish several articles, and then quit forever. Therefore, the author column does not provide enough power to predict the popularity of articles.

# Model

I would like to create a recurrent neural network (RNN) model to deal with the title and predict the popularity. 

The measure of popularity is defined as the adjusted logrithm of up-votes: the number of up-votes is added by one and then coverted to logrithm value. By the conclusion from network science, I assume the number of up-votes will increase expoentially according to its popularity, so logrithm transformation is preformed on the number in order to achieve better distribution of the values. 

Here I consider the title as string of words with logical meaning, and try to use the language to predict the popularity. First of all, I tokenize the character string into a list of words, splitting at space and punctuation. After this process, the original title is coverted into word lists, and all following steps are performed using the units of word. 

Then basd on the list of words, I used embedding method to covert the dictionary of words into dictionary of vectors. The embedding of words to build measure of distance within the vocabulary set (i.e. build connections among words), so that the RNN model could take word meaning into consideration.

To deal with the gradient vanishing / explosition problem, the gating method of LSTM is used. Here I build several recurent layers with LSTM to process the word list. Then dense layers are used to furthermore process the data set, until then, a single output is calculated as the popularity.

To deal with the potentially huge amount of data (over 100GB), iterative method is used. Each time only a part of data is read into memory, and the RNN model is trained or updated based on current chunk of data. With iterations over dataset, the model is trained on each part of the dataset and finally updated to fit the whole large dataset. 

# Coding

Firstly, I imported several python libraries including math, numpy, pandas, and tensorflow.
```python
# import packages
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
```

If extremely large datasets are needed, the Apache Spark might be used. Here it is not necessary to use Spark. 
```python
# import pyspark
# import findspark
# import os
# from pyspark import SparkContext
```

For later usage, I defined several constants: "filename" to store the location of dataset, "chunksize" to define the size of chunk to be read into memory in each iteration, "dictsize" to define the size of dictionary of often-used words, "maxlen" to define the maximum length of title to be considered in RNN model, and "embedsize" to define the embedding dimensions.
```python
# define parameters
filename = "Eluvio_DS_Challenge.csv"
chunksize = 100000
dictsize = 1000
maxlen = 30
embedsize = 5
```

Then a full dictionary is build by traversing the whole dataset, for simplicity just read the whole dataset or a fixed amount of lines in the whole dataset. After reading in the dataset, the dictionary tokenizer is fitted on it.
```python
# fulldf = pd.read_csv(filename, nrows=chunksize)
fulldf = pd.read_csv(filename)
# create tokenizer object (convert words to integers)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=dictsize)
tokenizer.fit_on_texts(fulldf["title"])
```

Here I use the RNN model with 1 hidden layer of embedding, 3 hidden layer of LSTM, 2 hidden layers of dense neurons. By testing several options, the activation functions are set to be "Sigmoid" or "ReLU", and the optimizer is set to be "Adam". (See the model summary in the output.txt file)
```python
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
```

Then the RNN model is trained on chunks of dataset iteratively. Here all text titles are converted to word lists of max length 30. Extra words in the end are removed if word list is too long, and zeros are added to the front if too short. In each of the interations, validation of size 5% is used. Batch of size 100 is trained each time and the chunk of dataset is trained repeated with 5 epochs. 
```python
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
```

Finally, the RNN model is used to predict the popularity on testing dataset created selected from each chunk of data. For simplicity, here I only use the latest data as my testing set, since it is meaningful to predict newly publshed articles.
```python
# evaluate RNN model
result = rnnmodel.evaluate(test_x, test_y)
print("MSE: ", result)

mean = sum(df_y)/len(df_y)
var = sum(np.square(np.add(df_y, - mean)))/len(df_y)
print("compare:", var)
# predict RNN model
# result = rnnmodel.predict(pred_x)
```

# Discussion 

Using MSE as the measure of loss, the final prediction has loss of 3.39, while the average total loss in the data set is 3.82. Therefore, the RNN model could indeed explan the popularity by using only the title information. That is to say, the RNN model indeed can be considered as a automatic prediting system, and can be regarded as a reference for popularity of articles. However, the loss itself is still large, and thus the model could still be improved.

One reason for the RNN model not performing as expected is that the dataset is too small to train a complicated RNN, while a too simple RNN model could not explan the data well. The number of up-votes is a result of complicated process, it depends on a lot of variables, and in order to predict it using machine learning, sufficiently large dataset is necessary. Therefore, if possible, more data should be put into model in order to have a better result. 

Another reason is that the prediction is purely based on title information. The information provided is insufficient, and more useful features of the news articles should be considered. For example, full text, publication platform, author's years of experience, historical events happend at each time, etc. Even for human, it is not possible to fully predict popularity only based on title, and more information must be provided. 





