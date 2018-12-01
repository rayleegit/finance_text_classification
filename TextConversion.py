
# coding: utf-8

# ## Create dataset

# ## Instructions:
# create testing folder in home (~/) directory
# download glove embeddings and place "glove.6B" folder in home~/ directory http://nlp.stanford.edu/data/glove.6B.zip

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


# In[162]:


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import model_from_json

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


# Word2Vec
from gensim.test.utils import common_texts, get_tmpfile # not working
from gensim.models import Word2Vec

import re
from collections import Counter
import os


from sklearn.model_selection import train_test_split



# ## Investigate ZIP file--why is filesize much lower, why is there so many NaN values

# In[90]:



# list_of_csv = os.listdir("/home/raymondleemids/testing/")


# In[91]:


# print(len(list_of_csv))
# # s/b ~1500


# In[92]:


# # home_dir = !~/testing
# # some files may be corrupted or format incorrect. only using 50 now

# # csv_list = []

# # for i in list_of_txt:
# #     csv_list.append(pd.read_csv('~/testing/%s' % i))

# # merged_csv = pd.concat(csv_list)



# csv_list = []


# for i in list_of_csv:
#     i = '~/testing/' + i
#     try:
#         csv_list.append(pd.read_csv(i))
#     except:
#         pass

# merged_csv = pd.concat(csv_list, ignore_index=True)


# merged_csv = merged_csv.drop(columns=['Close', 'Open', 'Day Change', 'Low', 'High', 'Overnight Change', 'Volume','Adj Close'])
# merged_csv.to_csv('final.csv')


# In[ ]:


# pd.read_csv('~/testing/%s' % 'PGR.csv')


# In[94]:


# merged_csv


# In[93]:


# merged_csv.shape


# In[95]:


# # merged_csv.groupby('Ticker')['Close'].count()
# merged_csv[merged_csv.Close_stock.isnull()].shape


# In[96]:


# nans = merged_csv[merged_csv.Close_stock.isnull()]


# In[97]:


# nans.to_csv('nans.csv', encoding='utf-8')


# In[98]:


# merged_csv[merged_csv.Close_stock.isnull()]['Ticker'].unique()


# In[99]:


# merged_csv.to_csv('merged.csv', encoding='utf-8')


# ## Skip to this part to load 'merged.csv'

# In[100]:


merged_csv = pd.read_csv('merged.csv')


# In[101]:


# len(csv_list)


# ## Moving forward with small subset of files because concat error

# In[ ]:


# # may remove this after fixing concat error
# csv_list = []

# for i in list_of_txt[:10]:
#     csv_list.append(pd.read_csv('~/testing/%s' % i))

# merged_csv = pd.concat(csv_list)


# In[ ]:


# merged_csv


# ## Clean and Tokenize Words

# In[102]:


# clean words 

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text

# vocabulary_size = 20000
# tokenizer = Tokenizer(num_words= vocabulary_size)
# tokenizer.fit_on_texts(df['text'])

# sequences = tokenizer.texts_to_sequences(df['text'])
# data = pad_sequences(sequences, maxlen=50)


# In[103]:


merged_csv = merged_csv.dropna()


# In[104]:


# tokenize words
merged_csv['8K_Content_cleaned'] = merged_csv['8K_Content'].map(lambda x: clean_text(x))


# In[105]:


merged_csv['8K_Content_cleaned'][:2]


# In[123]:


merged_csv.to_csv('merged.csv', encoding='utf-8')


# In[196]:


merged_csv = pd.read_csv('merged.csv')


# In[197]:


merged_csv.shape


# In[198]:


merged_csv = merged_csv.dropna()


# In[199]:


merged_csv.shape


# ## Add Price Movement label

# In[200]:


merged_csv['Overnight Change_stock percentage'].mean()


# In[201]:


merged_csv['Overnight Change_stock percentage'].std()


# In[202]:


# # values below mean + std/4
sum(merged_csv['Overnight Change_stock percentage'] > (merged_csv['Overnight Change_stock percentage'].mean() + merged_csv['Overnight Change_stock percentage'].std() / 4))


# In[203]:


# # values below mean - std/4
sum(merged_csv['Overnight Change_stock percentage'] < (merged_csv['Overnight Change_stock percentage'].mean() - merged_csv['Overnight Change_stock percentage'].std()/ 4))


# In[204]:


# total values
len(merged_csv['Overnight Change_stock percentage'])


# In[205]:


#TODO: create more robust UP/DOWN/STAY logic


# In[206]:


# UP/DOWN/STAY logic
merged_csv['movement'] = merged_csv['Overnight Change_stock percentage'] > 0


# In[207]:


# class balance is okay
sum(merged_csv['movement']) / len(merged_csv['movement'])


# In[208]:


merged_csv


# ## Create Embeddings

# In[209]:


# TODO: see if should increase string length
merged_csv['8K_Content_cleaned'][0]


# In[210]:


# average length of '8K_Content_cleaned'
merged_csv['8K_Content_cleaned_length'] = merged_csv['8K_Content_cleaned'].str.split(" ").str.len()


# In[211]:


merged_csv['8K_Content_cleaned_length'].mean()


# In[212]:


merged_csv['8K_Content_cleaned_length'].dropna().mean()


# In[213]:


# np.histogram(merged_csv['8K_Content_cleaned_length'].dropna())
# plt.pyplot.hist(merged_csv['8K_Content_cleaned_length'].dropna(), bins=50, range=[0,20000])


# In[214]:


# most records are under 3000 words

sum(merged_csv['8K_Content_cleaned_length'] < 3000) / len(merged_csv['8K_Content_cleaned_length'])


# In[ ]:


# TODO: update vocabulary_size with non-arbitary number

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(merged_csv['8K_Content_cleaned'])

sequences = tokenizer.texts_to_sequences(merged_csv['8K_Content_cleaned'])
data = pad_sequences(sequences, maxlen=3000)


# ## Split Dataset Train/Test

# In[ ]:


## Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    merged_csv['movement'], 
                                                    test_size=0.2, random_state=42)


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


# In[ ]:


# TODO: update arbitary parameters, such as input_length
# build network

model_lstm = Sequential()
model_lstm.add(Embedding(20000, 100, input_length=3000))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='softmax'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# train network

model_lstm.fit(X_train, y_train, validation_split=0.4, epochs=1)

# takes about 3 mins per epoch using 2/3 of full dataset
# takes about 3 hours per epoch using full dataset (less testing set) 


# In[ ]:


# TODO: create train/test/split

# test network

model_lstm.evaluate(X_test, y_test)


# In[ ]:


# predict

min(model_lstm.predict(X_test))


# In[ ]:


model_lstm.predict(X_test)


# In[ ]:


# save model

# serialize model to JSON
model_lstm_json = model_lstm.to_json()
with open("model_lstm.json", "w") as json_file:
    json_file.write(model_lstm_json)
# serialize weights to HDF5
model_lstm.save_weights("model_lstm.h5")
print("Saved model to disk")


# In[62]:


# load model

# # load json and create model
# json_file = open('model_lstm.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
# loaded_model.load_weights("model_lstm.h5")
# print("Loaded model from disk")


# In[63]:


# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(data, merged_csv['movement'], verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# ## LSTM with 1D Convolutional Layer

# In[117]:


def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, 100, input_length=3000))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='softmax'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv


# In[118]:


model_conv = create_conv_model()
model_conv.fit(data, merged_csv['movement'], validation_split=0.4, epochs = 1)


# In[119]:


model_conv.evaluate(data, merged_csv['movement'])


# In[120]:


min(model_conv.predict(data))


# In[121]:


model_conv.predict(data)


# In[122]:


# # save model

model_conv_json = model_conv.to_json()
with open("model_conv.json", "w") as json_file:
    json_file.write(model_conv_json)
# serialize weights to HDF5
model_conv.save_weights("model_conv.h5")
print("Saved model to disk")


# In[64]:


# # load model

# # load json and create model
# json_file = open('model_conv.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
# loaded_model.load_weights("model_conv.h5")
# print("Loaded model from disk")


# In[ ]:


# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(data, merged_csv['movement'], verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


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

