#!/usr/bin/env python
# coding: utf-8
import json

with open('/home/tanyixin/ISWC_2020/data/tags_occurrence.json') as tags:
    tags_occurrence = json.load(tags)

with open('/home/tanyixin/ISWC_2020/data/labels.txt') as f:
    lables = f.readlines()

# hypernym network construction    
hypernym = {}
for line in lables:
    line = line.strip().split(' ')
    hypernym[line[0]] = line[1].split(',')[-1]

union_network = {}
count = 0
for tag in tags_occurrence:
#     if count > 3:
#         break
    try:
        tag_concept = hypernym[tag]
    except:
        continue
    instance_list = []
    for item in tags_occurrence[tag]:
        try:
            instance_item = hypernym[item]
        except:
            continue
        if instance_item not in instance_list:
            instance_list.append(instance_item)
    if tag_concept in union_network:
        for word in instance_list:
            if word not in union_network[tag_concept]:
                union_network[tag_concept].append(word)
    else:
        union_network[tag_concept] = instance_list

centrality = []
for tag in union_network:
    centrality.append((tag, len(union_network[tag])))
    
result = sorted(centrality, key=lambda x: (-x[1],x[0]))

with open('/home/tanyixin/ISWC_2020/data/centrality.txt', 'w') as f:
    for pair in result:
        tag = pair[0]
        centrality_index = pair[1]/(len(result)-1)
        f.write(tag + ' ' + str(centrality_index) + '\n')