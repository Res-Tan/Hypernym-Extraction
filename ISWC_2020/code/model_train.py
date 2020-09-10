#!/usr/bin/env python
# coding: utf-8

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import Bidirectional
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.backend import manual_variable_initialization, manual_variable_initialization
from keras.models import Sequential
from keras.utils import np_utils, plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from tensorflow import nn as nn
import zipfile
import numpy as np
import time


# Pos tag corpus construction and one-hot representation

with open('/home/tanyixin/ISWC_2020/data/all_defintion_pos.txt', 'r') as f:
    corpus_chars = f.readlines()
    
corpus = []
for line in corpus_chars:
    text = line.strip().split(' ')[1:]
    for word in text:
        corpus.append(word)
        
corpus_index = list(set(corpus))
corpus_index.append('b')
vocab_size = len(corpus_index)
print(vocab_size) # 16

pos_index = {}
for pos in range(len(corpus_index)):
    pos_index[corpus_index[pos]] = pos
    
y_onehot = np_utils.to_categorical(np.array(range(len(corpus_index)), dtype='float64'))
vector_onehot = (y_onehot*0.98)+0.01

with open('/home/tanyixin/ISWC_2020/data/train_labels.txt', 'r') as f:
    line_list = f.readlines()
    
manual_tags_train = {}
for line in line_list:
    line = line.strip().split(' ')
    tag_name = line[0]
    tag_pos = line[1]
    manual_tags_train[tag_name] = tag_pos
    
with open('/home/tanyixin/ISWC_2020/data/candidate_nouns.txt', 'r') as f:
    line_list = f.readlines()
    
candidate_tags = {}
for line in line_list:
    line = line.strip().split(' ')
    candidate_tags[line[0]] = line[1:]
    
with open('/home/tanyixin/ISWC_2020/data/centrality.txt', 'r') as f:
    line_list = f.readlines()
    
centrality = {}
for line in line_list:
    line = line.strip().split(' ')
    centrality[line[0]] = float(line[1])


# trainset construction (window size = 9)
with open('/home/tanyixin/ISWC_2020/data/train_definition_pos.txt', 'r') as f:
    line_list = f.readlines()

x = []
y = []
print(len(line_list))

window_size = 9
for line in line_list:
    line = line.strip().split(' ')
    tag = line[0]
    try:
        manual_tags_train[tag]
        candidate_tags[tag]
    except:
        continue
    
    sentence = line[1:]
    index_nn_list = []
    for index in range(len(sentence)):
        if sentence[index] == 'NN':
            index_nn_list.append(index)
#     print(index_nn_list)
    score = []
    input_list = []
    
    target_word_index = 0
    target_word = manual_tags_train[tag].split(',')[-1]
    for index in range(len(candidate_tags[tag])):
        if candidate_tags[tag][index] == target_word:
            target_word_index = index
        
    count = 0
    for index_nn in index_nn_list:
        if index_nn - window_size < 0:
            prev_three_index = 0
        else:
            prev_three_index = index_nn - window_size
            
        if index_nn + window_size > len(sentence) - 1:
            next_three_index = len(sentence) - 1
        else:
            next_three_index = index_nn + window_size
            
#         print(prev_three_index)
#         print(next_three_index)
        prev_list = sentence[prev_three_index: index_nn]
        next_list = sentence[index_nn + 1: next_three_index + 1]
        
        if len(prev_list) < window_size:
            prev_list.insert(0, 'b')
            
        if len(next_list) < window_size:
            next_list.append('b')
        
#         print(prev_list)
#         print(next_list)
        prev_score = []
        next_score = []
        for i in range(window_size):
            try:
                prev_score.append(vector_onehot[pos_index[prev_list[i]]])
            except:
                prev_score.append(vector_onehot[pos_index['b']])
                
            try:
                next_score.append(vector_onehot[pos_index[next_list[i]]])
            except:
                next_score.append(vector_onehot[pos_index['b']])
              
        x.append(np.vstack((np.array(prev_score), np.array(next_score))))
        if count == target_word_index:
            y.append([0.99, 0.01])
        else:
            y.append([0.01, 0.99])
        count += 1
        
x_train = np.array(x)
y_train = np.array(y)
print(x_train.shape)
print(y_train.shape)


# model 1 for Pos context sequence learning
model1 = Sequential()
model1.add(Bidirectional(GRU(32, activation='tanh'), input_shape=(window_size*2, vocab_size)))
model1.add(Dropout(0.2))
model1.add(Dense(16))
model1.add(Activation('relu'))
model1.add(Dense(2))
model1.add(Activation('softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

start = time.time()
model1.fit(x_train, y_train, batch_size=32, epochs=80, verbose=0)
print(time.time() - start)

# refinement uion
with open('/home/tanyixin/ISWC_2020/data/train_definition_pos.txt', 'r') as f:
    line_list = f.readlines()

x = []
y = []
print(len(line_list))

for line in line_list:
    line = line.strip().split(' ')
    tag = line[0]
    try:
        manual_tags_train[tag]
        candidate_tags[tag]
    except:
        continue
    
    sentence = line[1:]
    index_nn_list = []
    for index in range(len(sentence)):
        if sentence[index] == 'NN':
            index_nn_list.append(index)
#     print(index_nn_list)
    score = []
    input_list = []
    
    target_word_index = 0
    target_word = manual_tags_train[tag].split(',')[-1]
    for index in range(len(candidate_tags[tag])):
        if candidate_tags[tag][index] == target_word:
            target_word_index = index
        
    count = 0
    for index_nn in index_nn_list:
        if index_nn - window_size < 0:
            prev_three_index = 0
        else:
            prev_three_index = index_nn - window_size
            
        if index_nn + window_size > len(sentence) - 1:
            next_three_index = len(sentence) - 1
        else:
            next_three_index = index_nn + window_size
            
#         print(prev_three_index)
#         print(next_three_index)
        prev_list = sentence[prev_three_index: index_nn]
        next_list = sentence[index_nn + 1: next_three_index + 1]
        
        if len(prev_list) < window_size:
            prev_list.insert(0, 'b')
            
        if len(next_list) < window_size:
            next_list.append('b')
        
#         print(prev_list)
#         print(next_list)
        prev_score = []
        next_score = []
        for i in range(window_size):
            try:
                prev_score.append(vector_onehot[pos_index[prev_list[i]]])
            except:
                prev_score.append(vector_onehot[pos_index['b']])
                
            try:
                next_score.append(vector_onehot[pos_index[next_list[i]]])
            except:
                next_score.append(vector_onehot[pos_index['b']])
              
        x_temp = np.vstack((np.array(prev_score), np.array(next_score)))
        x_temp = np.expand_dims(x_temp, axis=0)
        x_pred = model.predict(x_temp)
        pred = x_pred.tolist()[0][0]
        location = index_nn + 1 / len(sentence)
        candidate_nn = candidate_tags[tag][count]
        try:
            centrality_measure = centrality[candidate_nn]
        except:
            centrality_measure = 0
        capitalized = 0.01
        if candidate_nn[0].isupper():
            capitalized = 0.99
#         x_new = [pred, location, centrality_measure, capitalized]
        x_new = [pred, location, centrality_measure]
        x.append(x_new)
        if count == target_word_index:
            y.append([0.99, 0.01])
        else:
            y.append([0.01, 0.99])
        count += 1
x_train = np.array(x)
y_train = np.array(y)
print(x_train.shape)
print(y_train.shape)

# refinement model 2
model2 = Sequential()
model2.add(Dense(16, input_shape=(3,)))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

start = time.time()
model2.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)
print(time.time() - start)


# prediction (testset)
with open('/home/tanyixin/ISWC_2020/data/test_definition_pos.txt', 'r') as f:
    line_list = f.readlines()

# print(len(line_list))
fff = open('/home/tanyixin/ISWC_2020/data/predict/keras_ws9_refine.txt', 'w')
x = []
y = []

for line in line_list:
    line = line.strip().split(' ')
    tag = line[0]
    try:
        candidate_tags[tag]
    except:
        continue
    
    sentence = line[1:]
    index_nn_list = []
    x_refinement = []
    for index in range(len(sentence)):
        if sentence[index] == 'NN':
            index_nn_list.append(index)
#     print(index_nn_list)
    if len(index_nn_list) == 0:
        continue
    score = []
    input_list = []
        
    count = 0
    for index_nn in index_nn_list:
        if index_nn - window_size < 0:
            prev_three_index = 0
        else:
            prev_three_index = index_nn - window_size
            
        if index_nn + window_size > len(sentence) - 1:
            next_three_index = len(sentence) - 1
        else:
            next_three_index = index_nn + window_size
            
#         print(prev_three_index)
#         print(next_three_index)
        prev_list = sentence[prev_three_index: index_nn]
        next_list = sentence[index_nn + 1: next_three_index + 1]
        
        if len(prev_list) < window_size:
            prev_list.insert(0, 'b')
            
        if len(next_list) < window_size:
            next_list.append('b')
        
#         print(prev_list)
#         print(next_list)
        x_temp = []
        prev_score = []
        next_score = []
        for i in range(window_size):
            try:
                prev_score.append(vector_onehot[pos_index[prev_list[i]]])
            except:
                prev_score.append(vector_onehot[pos_index['b']])
                
            try:
                next_score.append(vector_onehot[pos_index[next_list[i]]])
            except:
                next_score.append(vector_onehot[pos_index['b']])
            
        x_temp.append(np.vstack((np.array(prev_score), np.array(next_score))))
#         x_test.append(np.array(x_temp).reshape(window_size*2, vocab_size))
        x_temp = np.array(x_temp).reshape(1, window_size*2, vocab_size)
        y_temp = model.predict(x_temp)
        pred = y_temp.tolist()[0][0]
        location = index_nn / len(sentence)
        candidate_nn = candidate_tags[tag][count]
        try:
            centrality_measure = centrality[candidate_nn]
        except:
            centrality_measure = 0
        capitalized = 0.01
        if candidate_nn[0].isupper():
            capitalized = 0.99
#         x_ref = [pred, location, centrality_measure, capitalized]
        x_ref = [pred, location, centrality_measure]
#         print(x_ref)
        x_refinement.append(x_ref)
        count += 1
    x_refinement = np.array(x_refinement)
#     print(x_refinement)
#     print(tag)
#     print(candidate_tags[tag])
    y_ref = model2.predict(x_refinement)
#     print(y_ref)
    tag_index = y_ref[:,0].argmax()
    tag_predict = candidate_tags[tag][tag_index]
#     print(tag_predict)
    fff.write(tag + ' ' + tag_predict + '\n')
fff.close()

# predict all defintions
with open('/home/tanyixin/ISWC_2020/data/all_defintion_pos.txt', 'r') as f:
    all_definitions = f.readlines()

fff = open('/home/tanyixin/ISWC_2020/data/predict/all_predict.txt', 'w')
x = []
y = []

for line in all_definitions:
    line = line.strip().split(' ')
    tag = line[0]
    try:
        candidate_tags[tag]
    except:
        continue
    
    sentence = line[1:]
    index_nn_list = []
    x_refinement = []
    for index in range(len(sentence)):
        if sentence[index] == 'NN':
            index_nn_list.append(index)
#     print(index_nn_list)
    if len(index_nn_list) == 0:
        continue
    score = []
    input_list = []
        
    count = 0
    for index_nn in index_nn_list:
        if index_nn - window_size < 0:
            prev_three_index = 0
        else:
            prev_three_index = index_nn - window_size
            
        if index_nn + window_size > len(sentence) - 1:
            next_three_index = len(sentence) - 1
        else:
            next_three_index = index_nn + window_size
            
#         print(prev_three_index)
#         print(next_three_index)
        prev_list = sentence[prev_three_index: index_nn]
        next_list = sentence[index_nn + 1: next_three_index + 1]
        
        if len(prev_list) < window_size:
            prev_list.insert(0, 'b')
            
        if len(next_list) < window_size:
            next_list.append('b')
        
#         print(prev_list)
#         print(next_list)
        x_temp = []
        prev_score = []
        next_score = []
        for i in range(window_size):
            try:
                prev_score.append(vector_onehot[pos_index[prev_list[i]]])
            except:
                prev_score.append(vector_onehot[pos_index['b']])
                
            try:
                next_score.append(vector_onehot[pos_index[next_list[i]]])
            except:
                next_score.append(vector_onehot[pos_index['b']])
            
        x_temp.append(np.vstack((np.array(prev_score), np.array(next_score))))
#         x_test.append(np.array(x_temp).reshape(window_size*2, vocab_size))
        x_temp = np.array(x_temp).reshape(1, window_size*2, vocab_size)
        y_temp = model.predict(x_temp)
        pred = y_temp.tolist()[0][0]
        location = index_nn / len(sentence)
        candidate_nn = candidate_tags[tag][count]
        try:
            centrality_measure = centrality[candidate_nn]
        except:
            centrality_measure = 0
        capitalized = 0.01
        if candidate_nn[0].isupper():
            capitalized = 0.99
#         x_ref = [pred, location, centrality_measure, capitalized]
        x_ref = [pred, location, centrality_measure]
#         print(x_ref)
        x_refinement.append(x_ref)
        count += 1
    x_refinement = np.array(x_refinement)
#     print(x_refinement)
#     print(tag)
#     print(candidate_tags[tag])
    y_ref = model2.predict(x_refinement)
#     print(y_ref)
    tag_index = y_ref[:,0].argmax()
    tag_predict = candidate_tags[tag][tag_index]
#     print(tag_predict)
    fff.write(tag + ' ' + tag_predict + '\n')
fff.close()