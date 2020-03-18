# NLP Challenge by Jiayi Fu

# Introduction

This is a NLP Code Chanllenge project on a dataset about titles and votes. The data set containing 522278 rows following 8 columns: time_created, date_created, up_votes, down_votes, title, over_18, author, category.

Based on this dataset, the problem I came up with is to predict the number of up-votes purely based on the title. I would like to forecast the total number of up-votes when a news article is published long enough. This kind o fproblem is meaninful, beacuse for example, news platform could find potential popular articles on the front page and attract more customers. 

# Notes on Backgroud

By exploratory analysis, I ignored other information here for simplicity: 

Firstly, the date / time created column is ignored. Beacuse when the time passed is long enough (for example, in this data set, all articles are published at least several years ago), the total number of up-votes is amost stable and will not change much anymore. Readers are interested in newly published news instead of news of several years old. This is also justifyed by the conclusion from graph theory with long term adjustment, which is a reasonable model to discribe the model of news / publishment. 

The number down-votes are also ignored. Since most of the articles have 0 downvotes, it is more efficient to only look at the up-votes. If necessary in the future, the difference or adjusted ratio (avoid dividing by 0) of up / down votes could be used. It is also possible to use scaled ratings as dependent variable instead of just up-votes.

I also ignored the over 18 and category information. Beacuse of the prevalent "False" in over_18 column and "worldnews" in category column, these two columns do not provide significant enough information to my analysis. For simplicity of the model that concentrated on NLP, these two columns are dropped.

Author information is kind of unstable over long time, although it is true that there might be some popular authors that always creates articles with great amount of up-votes. According to the data set, the authors look like users instead of media companies, which means they are more likely to be come-and-go writters over long span of time, instead of reliable news soures. It will take too much time and space to deal with the author information, espcially when many authors may suddenly jump in, only publish several articles, and then quit forever. Therefore, the author column does not provide enough power to predict the popularity of articles.

# Model




