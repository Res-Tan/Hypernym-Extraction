#!/usr/bin/env python
# coding: utf-8
with open('/home/tanyixin/ISWC_2020/data/labels.txt', 'r') as f:
    label_pairs = f.readlines()


# trainset and testset
labels = {}
train_set = {}
test_set = {}
count = 0
for label_pair in label_pairs:
    label_pair = label_pair.strip().split(' ')
    labels[label_pair[0]] = label_pair[1]
    if count < 3000:
        train_set[label_pair[0]] = label_pair[1]
    else:
        test_set[label_pair[0]] = label_pair[1]
    count += 1


# Specific PoS Tag reservation
nn = ['NN', 'NNP', 'NNS']
reserve_pos = ['DT', 'EX', 'IN', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

with open('/home/tanyixin/ISWC_2020/data/tags_clean_pos.txt', 'r') as f:
    line_list = f.readlines()

train_definition = {}
test_definition = {}
count = 0
for line in line_list:
    line = line.strip().split(' ')
    tag = line[0]
    definition = line[1:]
    if tag not in labels:
        continue
    
    filter_definition = []
    for pos in definition:
        if pos in reserve_pos:
            filter_definition.append(pos)
        elif pos in nn:
            filter_definition.append('NN')
        else:
            continue
            
    if count < 3000:
        train_definition[tag] = filter_definition
    else:
        test_definition[tag] = filter_definition
        
    count += 1


f1 = open('/home/tanyixin/ISWC_2020/data/train_definition_pos.txt', 'w')

for tag in train_definition:
    defintion_pos = ' '.join(x for x in train_definition[tag])
    f1.write(tag + ' ' + defintion_pos + '\n')
    
f1.close()

f2 = open('/home/tanyixin/ISWC_2020/data/test_definition_pos.txt', 'w')

for tag in test_definition:
    defintion_pos = ' '.join(x for x in test_definition[tag])
    f2.write(tag + ' ' + defintion_pos + '\n')
    
f2.close()

f3 = open('/home/tanyixin/ISWC_2020/data/train_labels.txt', 'w')

for tag in train_set:
    f3.write(tag + ' ' + train_set[tag] + '\n')
    
f3.close()

f4 = open('/home/tanyixin/ISWC_2020/data/test_labels.txt', 'w')

for tag in test_set:
    f4.write(tag + ' ' + test_set[tag] + '\n')
    
f4.close()

# count total words
with open('/home/tanyixin/ISWC_2020/data/tags_clean.txt', 'r') as f:
    line_list = f.readlines()

corpurs = []
for line in line_list[:4786]:
    line = line.strip().split(' ')
    for word in line[1:]:
        corpus.append(word)

corpurs = list(set(corpus))

print(len(corpurs)) # 9921

all_definition = {}
for line in line_list:
    line = line.strip().split(' ')
    tag = line[0]
    definition = line[1:]
    
    filter_definition = []
    for pos in definition:
        if pos in reserve_pos:
            filter_definition.append(pos)
        elif pos in nn:
            filter_definition.append('NN')
        else:
            continue
            
    all_definition[tag] = filter_definition

with open('/home/tanyixin/ISWC_2020/data/all_defintion_pos.txt', 'w') as f:
    for tag in all_definition:
        defintion_pos = ' '.join(x for x in all_definition[tag])
        f.write(tag + ' ' + defintion_pos + '\n')