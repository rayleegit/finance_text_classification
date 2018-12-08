
# coding: utf-8

# ## Create dataset

# ## Instructions:
# create testing folder in home (~/) directory 
# 
# download glove embeddings and place "glove.6B" folder in home~/ directory http://nlp.stanford.edu/data/glove.6B.zip
# 
# also, increase the speed of your instance (i used 16 CPUs and it took 6 hours to train 1 epoch of 3,000 word sequences)

# In[19]:


# !pip install keras
# !pip install tensorflow
# !pip install plotly
# !pip install gensim
# !pip install Word2Vec
# !pip install get_tmpfile
# !pip install gensim.test.utils
# !pip install boto
# !pip install google-compute-engine


# In[3]:


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
from keras.preprocessing.text import text_to_word_sequence






## Plot
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import matplotlib as plt
# import matplotlib.pyplot

# NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import SnowballStemmer


# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from numpy import util


# Word2Vec
from gensim.test.utils import common_texts, get_tmpfile # not working
from gensim.models import Word2Vec

import re
from collections import Counter
import os








# ## [Skip this section] Split Dataset Train/Test 

# In[5]:


#load and read final.csv
# final_csv.to_csv('final.csv')
# final_csv = pd.read_csv('final.csv')


# In[4]:


# final_csv[:5]


# In[6]:


### loaded into csv
# # convert integers to dummy variables (i.e. one hot encoded)
# encoder = LabelBinarizer()
# sparse_labels = encoder.fit_transform(final_csv['stock_overnight_movement_tag'])


# In[7]:


### loaded into csv
# encoder.classes_


# In[8]:


### loaded into csv
# # create sparse label columns
# sparse_labels = pd.DataFrame(data=sparse_labels,columns=['down', 'stay', 'up'])


# In[9]:


### loaded into csv
# final_csv['down'] = sparse_labels['down']
# final_csv['stay'] = sparse_labels['stay']
# final_csv['up'] = sparse_labels['up']


# In[21]:


# randomize dataset order 
# final_csv = shuffle(final_csv)


# In[22]:


# # output shuffled.csv
# final_csv.to_csv('final_shuffled.csv')


# ## Load final_shuffled.csv

# In[23]:


# load final_shuffled.csv
# final_csv = pd.read_csv('final_shuffled.csv')


# In[444]:


# # filter final_csv to only include 'out of market'
# backup_csv = final_csv # backup in case i want to revert back
# final_csv = final_csv[final_csv['Time_of_day'] == 'out_of_market']


# In[ ]:


# downsample
# final_csv = pd.concat([final_csv[final_csv['down'] == 1], 
#      final_csv[final_csv['stay'] == 1].iloc[:10750], 
#     final_csv[final_csv['up'] == 1]])


# In[ ]:


# final_csv.to_csv('final_downsampled.csv')


# In[4]:


final_csv = pd.read_csv('final_shuffled.csv')


# In[5]:


final_csv


# ## Create '8K_Content' sequences

# In[24]:


# tokenize text. update this when you update sequence length

tokenizer = Tokenizer()
tokenizer.fit_on_texts(final_csv['8K_Content'])


seq_len = 300 # set sequence length
sequences = tokenizer.texts_to_sequences(final_csv['8K_Content'])
data = pad_sequences(sequences, 
                     maxlen=seq_len, padding='post', 
                     truncating='post') # takes about 5-10 mins


# In[25]:


# vocab size
vocabulary_size = max(tokenizer.word_index.values())
vocabulary_size


# In[26]:


final_csv['8K_Content_sequences'] = data.tolist()


# In[154]:


final_csv


# In[171]:


# sum(final_csv['down'])


# In[172]:


# sum(final_csv['stay']) # there are 85,956 'stay' classes. will modify this to 10,750 to match down and up


# In[158]:


# sum(final_csv['up'])


# In[170]:


# final_csv[final_csv['stay'] == 1][:10750]


# In[174]:


# downsample_csv = pd.concat([final_csv[final_csv['down'] == 1], 
#      final_csv[final_csv['stay'] == 1][:10750], 
#     final_csv[final_csv['up'] == 1]])


# In[175]:


# sum(downsample_csv['down'])


# In[176]:


# sum(downsample_csv['stay'])


# In[177]:


# sum(downsample_csv['up'])


# In[ ]:


downsample_csv = pd.concat([final_csv[final_csv['down'] == 1], 
     final_csv[final_csv['stay'] == 1][:10750], 
    final_csv[final_csv['up'] == 1]])


# In[ ]:


final_csv = downsample_csv


# In[ ]:


final_csv['8K_Content_sequences'] = final_csv['8K_Content_sequences'].tolist()


# In[27]:


# 02-08 train. 09-10 develoment, 11-12 test
# X_train = final_csv['8K_Content_cleaned'][final_csv['Year'] <= 2008]
# X_dev = final_csv['8K_Content_cleaned'][(final_csv['Year'] >= 2009) & 
#                                        (final_csv['Year'] <= 2010)]
# X_test = final_csv['8K_Content_cleaned'][final_csv['Year'] >= 2011]

X_train = final_csv['8K_Content_sequences'][final_csv['Year'] <= 2008]
X_dev = final_csv['8K_Content_sequences'][(final_csv['Year'] >= 2009) & 
                                       (final_csv['Year'] <= 2010)]
X_test = final_csv['8K_Content_sequences'][final_csv['Year'] >= 2011]

y_train = final_csv[['down','stay','up']][final_csv['Year'] <= 2008]
y_dev = final_csv[['down','stay','up']][(final_csv['Year'] >= 2009) & 
                                       (final_csv['Year'] <= 2010)]
y_test = final_csv[['down','stay','up']][final_csv['Year'] >= 2011]


# ## Build and Train LSTM Model

# In[ ]:


# TODO: add Tensorboard

# reload(rnnlm)

# TF_GRAPHDIR = "/tmp/w266_project/finance_text_graph"

# # Clear old log directory.
# shutil.rmtree(TF_GRAPHDIR, ignore_errors=True)

# lm = rnnlm.RNNLM(V=10000, H=200, num_layers=2)
# lm.BuildCoreGraph()
# lm.BuildTrainGraph()
# lm.BuildSamplerGraph()

# summary_writer = tf.summary.FileWriter(TF_GRAPHDIR, lm.graph)


# In[508]:


# TODO: update arbitary parameters, such as input_length
# build network

model_lstm = Sequential()
model_lstm.add(Embedding(vocabulary_size, 100, input_length=seq_len))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(3, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[493]:


# np.array(X_train[:5].values.tolist()).shape


# In[497]:


# np.array(y_train[:5].values.tolist()).shape


# In[499]:


# y_train[:5].values.shape


# In[506]:


# np.array(y_train[:5].values).shape


# In[513]:


# train network

# model_lstm.fit(np.array(X_train[:5].values.tolist()), np.array(y_train[:5].values.tolist()), validation_split=0.4, epochs=5)

model_lstm.fit(np.array(X_train.values.tolist()), 
               np.array(y_train.values.tolist()), 
               validation_data=(np.array(X_dev.values.tolist()), 
                                np.array(y_dev.values.tolist())), 
               epochs=5)

#TODO: change validation split to validation_data

# takes about 3 mins per epoch using 2/3 of full dataset
# takes about 3 hours per epoch using full dataset (less testing set) 


# In[517]:


# test network

model_lstm.evaluate(np.array(X_test.values.tolist()), 
                    np.array(y_test.values.tolist()))
                    


# In[518]:


# predict
model_lstm_preds = model_lstm.predict(np.array(X_test.values.tolist()))


# In[523]:


# display prediction probs
# model_lstm_preds[:5]


# In[197]:


# set model # for saving the model and weights to json and h5
model_num = 4


# In[552]:


# predictions to csv
np.savetxt('predictions_lstm_%d.csv' %model_num, model_lstm_preds, delimiter=',')
print("Saved predictions to disk: ", 'predictions_lstm_%d.csv' %model_num)


# In[ ]:


# # history to csv
# np.savetxt('history_lstm_%d.csv' %model_num, history_lstm.history, delimiter=',')
# print("Saved history to disk: ", 'history_lstm_%d.csv' %model_num)


# In[532]:


# display prediction
np.argmax(model_lstm_preds, axis=1)


# In[536]:


# display actuals
np.argmax(np.array(y_test), axis=1)


# In[540]:


# prediction counts ['down','stay','up']
np.bincount(np.argmax(model_lstm_preds, axis=1))


# In[544]:


# actual counts ['down', stay', 'up']
np.bincount(np.argmax(np.array(y_test), axis=1))


# In[550]:


# save model

# serialize model to JSON
model_lstm_json = model_lstm.to_json()
with open("model_lstm_%d.json" %model_num, "w") as json_file:
    json_file.write(model_lstm_json)
# serialize weights to HDF5
model_lstm.save_weights("model_lstm_%d.h5" %model_num)
print("Saved model to disk: ", 'model_lstm_%d.json and model_lstm_%d.h5' %(model_num, model_num))


# In[ ]:


# load model
# note: latest file updated 11/30. 3,000 word sequence.

# load json and create model
json_file = open('model_lstm_%d.json' %model_num, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_lstm_%d.h5" %model_num)
print("Loaded model from disk: ", 'model_lstm_%d.json and model_lstm_%d.h5' %(model_num, model_num))


# In[266]:


# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(np.array(X_test.values.tolist()), 
                              np.array(y_test.values.tolist()), verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[267]:


# loaded_model.predict(X_test)


# ## LSTM with 1D Convolutional Layer

# In[117]:


def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, 100, input_length=seq_len))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(3, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv


# In[118]:


model_conv = create_conv_model()
model_conv.fit(np.array(X_train.values.tolist()), 
               np.array(y_train.values.tolist()), 
               validation_data=(np.array(X_dev.values.tolist()), 
                                np.array(y_dev.values.tolist())), 
               epochs=5)


# In[119]:


model_conv.evaluate(np.array(X_train.values.tolist()), 
               np.array(y_train.values.tolist()))


# In[120]:


# predict
model_conv_preds = model_conv.predict(np.array(X_test.values.tolist()))


# In[ ]:


# save predictions to csv
np.savetxt('predictions_conv_%d.csv' %model_num, model_conv_preds, delimiter=',')
print("Saved predictions to disk: ", 'predictions_conv_%d.csv' %model_num)


# In[ ]:


# # save history to csv
# np.savetxt('history_conv_%d.csv' %model_num, history_conv.history, delimiter=',')
# print("Saved history to disk: ", 'history_conv_%d.csv' %model_num)


# In[122]:


# # save model

model_conv_json = model_conv.to_json()
with open("model_conv_%d.json" %model_num, "w") as json_file:
    json_file.write(model_conv_json)
# serialize weights to HDF5
model_conv.save_weights("model_conv_%d.h5" %model_num)
print("Saved model to disk:", 'model_conv_%d.json and model_conv_%d.h5' %(model_num, model_num))


# In[268]:


# load model

# load json and create model
json_file = open('model_conv_%d.json' %model_num, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_conv_%d.h5" %model_num)
print("Loaded model from disk: ", 'model_conv_%d.json and model_conv_%d.h5' %(model_num, model_num))


# In[269]:


# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(np.array(X_test.values.tolist()), 
                              np.array(y_test.values.tolist()), verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[1]:





# In[270]:


# loaded_model.predict(X_test)


# ## Use pre-trained Glove word embeddings

# In[29]:


# # load GLOVE embeddings
# embeddings_index = dict()
# f = open('~/glove.6B/glove.6B.100d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))


# In[114]:


# !cd ~/glove.6B


# In[ ]:


# embeddings_index


# In[ ]:


# # create a weight matrix for words in training docs
# embedding_matrix = np.zeros((vocabulary_size, 100))
# for word, index in tokenizer.word_index.items():
#     if index > vocabulary_size - 1:
#         break
#     else:
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[index] = embedding_vector


# In[ ]:


# # LSTM and CNN model
# model_glove = Sequential()
# model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
# model_glove.add(Dropout(0.2))
# model_glove.add(Conv1D(64, 5, activation='relu'))
# model_glove.add(MaxPooling1D(pool_size=4))
# model_glove.add(LSTM(100))
# model_glove.add(Dense(1, activation='softmax'))
# model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# # train model
# model_glove.fit(data, merged_csv['movement'], validation_split=0.4, epochs=3)


# In[ ]:


# # evaluate model
# model_glove.evaluate(data, merged_csv['movement'])


# ## Use Word2Vec to train word embeddings on corpus

# In[ ]:


# from gensim.models import Word2Vec
# import nltk
# # nltk.download('punkt')


# In[ ]:


# # tokenize financial data
# merged_csv['tokenized'] = merged_csv.apply(lambda row : nltk.word_tokenize(row['8K_Content_cleaned']), axis=1)


# In[ ]:


# merged_csv.head()


# In[ ]:


# #train
# model_w2v = Word2Vec(merged_csv['tokenized'], size=100)


# In[ ]:


# X = model_w2v[model_w2v.wv.vocab]


# In[ ]:


# X.shape


# In[ ]:


# # Create LSTM and CNN model

# model_w2v = Sequential()
# model_w2v.add(Embedding(1925, 100, input_length=50, weights=[X], trainable=False))
# model_w2v.add(Dropout(0.2))
# model_w2v.add(Conv1D(64, 5, activation='relu'))
# model_w2v.add(MaxPooling1D(pool_size=4))
# model_w2v.add(LSTM(100))
# model_w2v.add(Dense(1, activation='softmax'))
# model_w2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# # train model
# model_w2v.fit(data, merged_csv['movement'], validation_split=0.4, epochs=3)


# ## Plot Word Vectors Using PCA

# In[300]:


# from sklearn.decomposition import TruncatedSVD


# In[301]:


# tsvd = TruncatedSVD(n_components=5, n_iter=10)
# result = tsvd.fit_transform(X)


# In[302]:


# result.shape


# In[304]:


# tsvd_word_list = []
# words = list(model_w2v.wv.vocab)
# for i, word in enumerate(words):
#     tsvd_word_list.append(word)


# In[309]:


# len(words)


# In[311]:


# tsvd_word_list = []
# words = list(model_w2v.wv.vocab)
# for i, word in enumerate(words):
#     tsvd_word_list.append(word)

# trace = go.Scatter(
#     x = result[0:len(words), 0], 
#     y = result[0:len(words), 1],
#     mode = 'markers',
#     text= tsvd_word_list[0:len(words)]
# )

# layout = dict(title= 'SVD 1 vs SVD 2',
#               yaxis = dict(title='SVD 2'),
#               xaxis = dict(title='SVD 1'),
#               hovermode= 'closest')

# fig = dict(data = [trace], layout= layout)
# py.iplot(fig)


# ## Visualize word embeddings

# In[247]:


# lstm_embds = model_lstm.layers[0].get_weights()[0]


# In[248]:


# conv_embds = model_conv.layers[0].get_weights()[0]


# In[249]:


# glove_emds = model_lstm.layers[0].get_weights()[0]


# In[250]:


# word_list = []
# for word, i in tokenizer.word_index.items():
#     word_list.append(word)


# In[251]:


# def plot_words(data, start, stop, step):
#     trace = go.Scatter(
#         x = data[start:stop:step,0], 
#         y = data[start:stop:step, 1],
#         mode = 'markers',
#         text= word_list[start:stop:step]
#     )
#     layout = dict(title= 't-SNE 1 vs t-SNE 2',
#                   yaxis = dict(title='t-SNE 2'),
#                   xaxis = dict(title='t-SNE 1'),
#                   hovermode= 'closest')
#     fig = dict(data = [trace], layout= layout)
#     py.iplot(fig)


# In[252]:


# # LSTM embeddings 

# lstm_tsne_embds = TSNE(n_components=2).fit_transform(lstm_embds)


# In[254]:


# plot_words(lstm_tsne_embds, 0, 2000, 1)


# In[253]:


# # CNN + LSTM
# conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)


# In[ ]:


# plot_words(conv_tsne_embds, 0, 2000, 1)


# In[255]:


# # Glove

# glove_tsne_embds = TSNE(n_components=2).fit_transform(glove_emds)


# In[256]:


# plot_words(glove_tsne_embds, 0, 2000, 1)

