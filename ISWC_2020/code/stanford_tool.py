#!/usr/bin/env python
# coding: utf-8

from stanfordcorenlp import StanfordCoreNLP
import logging
import json

nlp = StanfordCoreNLP('http://localhost', 9001, timeout=30000)

# Stanford test
text = 'search Questions about search algorithm mechanics and implementation'
nlp.pos_tag(text)
nlp.pos_tag('commercial general-purpose statistical software'.lower())
nlp.dependency_parse(text)
nlp.annotate(text)


nn = ['NN', 'NNP', 'NNS']

with open('/home/tanyixin/ISWC_2020/data/tags_clean.txt', 'r') as f:
    raw_list = f.readlines()
    
f1 = open('/home/tanyixin/ISWC_2020/data/candidate_nouns.txt', 'w')

with open('/home/tanyixin/ISWC_2020/data/tags_clean_pos.txt', 'w') as f:
    for line in raw_list:
        line = line.strip().split(' ')
        if len(line) <= 2:
            continue
        pos_definition = line[0]
        candidate = line[0]
        definition = ' '.join(x for x in line[1:])
        pos_tag = nlp.pos_tag(definition)
        
        for token in pos_tag:
            pos_definition = pos_definition + ' ' + token[1]
            if token[1] in nn:
                candidate = candidate + ' ' + token[0]
            
        f.write(pos_definition + '\n')
        f1.write(candidate + '\n')
        
f1.close()