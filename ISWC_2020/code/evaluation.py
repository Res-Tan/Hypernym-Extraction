#!/usr/bin/env python
# coding: utf-8

import numpy as np

with open('/home/tanyixin/ISWC_2020/data/candidate_nouns.txt', 'r') as f:
    line_list = f.readlines()
    
candidate_tags = {}
for line in line_list:
    line = line.strip().split(' ')
    candidate_tags[line[0]] = line[1:]
    
with open('/home/tanyixin/ISWC_2020/data/test_labels.txt', 'r') as f:
    manual_list = f.readlines()
    
manual_dict = {}
for line in manual_list:
    line = line.strip().split(' ')
    manual_dict[line[0]] = line[1]
    
with open('/home/tanyixin/ISWC_2020/data/predict/keras_ws9_refine.txt', 'r') as f:
    machine_list = f.readlines()
    
machine_dict = {}
for line in machine_list:
    line = line.strip().split(' ')
    if line[0] in machine_dict:
        print(line[0])
    machine_dict[line[0]] = line[1]

# non-defintion indentified by LSTM+CRF    
with open('/home/tanyixin/data_tag/paperuse/spamtest7', 'r') as f:
    spam_tag = f.readlines()
    
spam = []
for tag in spam_tag:
    spam.append(tag.strip())


right_predict = []
manual_error = []
machine_miss = []
error = []
for tag in machine_dict:
    if tag in spam:
        manual_error.append(tag)
        continue
    
    try:
        manual_dict[tag]
        if ',' in manual_dict[tag]:
            if machine_dict[tag].lower() in manual_dict[tag].split(','):
                right_predict.append(1)
            else:
                right_predict.append(0)
                error.append(tag)
        else:
            if machine_dict[tag].lower() == manual_dict[tag]:
                right_predict.append(1)
            else:
                right_predict.append(0)
                error.append(tag)
    except:
        machine_miss.append(tag)
        continue
            
print('manual : %d' % (len(manual_dict)))
print('machine : %d' % (len(machine_dict)))
# print('spam : %d' % (len(spam)))

score = np.array(right_predict)
print('miss predict : %d' % (len(machine_miss)+len(manual_dict)-len(machine_dict)))
print('manual error : %d' % (len(manual_error)))
print('right : %d' % (score.sum()))
print('error : %d' % (score.size - score.sum()))
print('machine miss : %d' % (len(machine_miss)))
acc = score.sum()/score.size
print('precision : %f' % (acc))
rec = score.sum()/len(manual_dict)
print('recall : %f' % (rec))
print('F1 : %f' % ((2*acc*rec)/(rec + acc)))