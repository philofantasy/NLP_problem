# NLP Challenge by Jiayi Fu

# Introduction

This is a NLP Code Chanllenge project on a dataset about titles and votes. The data set containing 522278 rows following 8 columns: time_created, date_created, up_votes, down_votes, title, over_18, author, category.

Based on this dataset, the problem I came up with is to predict the popularity purely based on the title. I would like to forecast the total number of up-votes when a news article is published long enough. This kind o fproblem is meaninful, beacuse for example, news platform could find potential popular articles on the front page and attract more customers. 

I used recurrent neural network (RNN) model to build the complicated regression / prediciton framework, and uses Python 3 with Tensorflow package to implement the method and finished model evaluation. To deal with huge size of data, only a part of data is read into memory and the model is trained gradually in iterations.

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

To deal with the potentially huge amount of data (over 100GB), iterative method is used. Each time only a part of data is read into memory, and the RNN model is trained or updated based on current batch of data. With iterations over dataset, the model is trained on each part of the dataset and finally updated to fit the whole large dataset. 

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

If extremely large datasets are needed, the Apache Spark might be used. Here it is not necessary
```ptyhon
# import pyspark
# import findspark
# import os
# from pyspark import SparkContext
```



