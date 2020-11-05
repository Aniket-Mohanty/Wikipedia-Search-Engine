#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
# import linecache
import linereader as lr
import re
from collections import *
import time
import sys

import numpy as np

import Stemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 


# In[2]:


index_path = './final_index1/'


# In[3]:


stemmer = Stemmer.Stemmer('english')


# In[4]:


start = time.perf_counter()
file = open(index_path + 'word_positions.pickle','rb')
word_positions             =  pickle.load(file)
file.close()
file = open(index_path + 'title_position.pickle','rb')
title_list  =  pickle.load(file)
file.close()

title_cache = lr.dopen(index_path + 'title.txt')
body_cache = lr.dopen(index_path + 'body.txt')
category_cache = lr.dopen(index_path + 'category.txt')
infobox_cache = lr.dopen(index_path + 'infobox.txt')
reference_cache = lr.dopen(index_path + 'reference.txt')
external_cache = lr.dopen(index_path + 'external.txt')

cache_list = {'t':title_cache,'b':body_cache,'c':category_cache,'i':infobox_cache,'r':reference_cache,'e':external_cache}

end = time.perf_counter()
print('Overhead time: ',end-start)


# In[5]:


def getline(file_manager,line_num):
    
    if(line_num == 1):
        return file_manager.getline(1)
    elif(line_num > 1):
        return file_manager.getlines(line_num-1,line_num)[1]


# In[6]:


def mergePost(post1, post2):

    ptr1 = 0
    ptr2 = 0
    
    p1_len = len(post1)
    p2_len = len(post2)
 
    docs = []
 
    while ((ptr1 < p1_len) and (ptr2 < p2_len)):

        if(post2[ptr2][0] < post1[ptr1][0]):   
            
            while((ptr2 < p2_len) and (post2[ptr2][0] < post1[ptr1][0])): ptr2 += 1
 
        elif(post1[ptr1][0] < post2[ptr2][0]): 
            
            while((ptr1 < p1_len) and (post1[ptr1][0] < post2[ptr2][0])): ptr1 += 1
 
        else:
            new_score = (post1[ptr1][1] + post2[ptr2][1])
            docs.append((post1[ptr1][0], new_score))
 
            ptr1 += 1
            ptr2 += 1

    return docs


# In[7]:


def unionPost(post1, post2):
    
    ptr1 = 0
    ptr2 = 0
    
    p1_len = len(post1)
    p2_len = len(post2)
    
    docs = []
    
    while ((ptr1 < p1_len) and (ptr2 < p2_len)):
        
        if(post1[ptr1][0] < post2[ptr2][0]):
            
            docs.append(post1[ptr1])
            ptr1 += 1
        
        elif(post1[ptr1][0] > post2[ptr2][0]):
            
            docs.append(post2[ptr2])
            ptr2 += 1
        
        else:
            
            docs.append(post1[ptr1])
            ptr1,ptr2 = ptr1+1,ptr2+1
            
    while (ptr1 < p1_len):
        
        docs.append(post1[ptr1])
        ptr1 += 1
    
    while (ptr2 < p2_len):
        
        docs.append(post2[ptr2])
        ptr2 += 1
        
    return docs


# In[8]:


def mergeAll(querio, fields, cache_list = cache_list, word_positions = word_positions):
    
    fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}

    result = []
    flag = 0

    for field,quer in zip(fields,querio):

        query_words = quer.strip().split(' ')

        for word in query_words:

            word = stemmer.stemWord(word.lower())

            if(word not in stop_words):
                if(word in word_positions[field]):
                    file = cache_list[field]
                    docs = getline(file,word_positions[field][word])[:-1]
                    docs = docs.split('|')
                    docs = [list(map(float,doc.split(':'))) for doc in docs]

                    if(not flag): 
                        result = docs
                        flag = 1

                    else: result = mergePost(result,docs)

    return result


# In[9]:


def unionAll(querio, fields, cache_list = cache_list, word_positions = word_positions):
    
    fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}

    result = []

    for field,quer in zip(fields,querio):

        query_words = quer.strip().split(' ')
        word_result = []
        
        for word in query_words:

            word = stemmer.stemWord(word.lower())

            if(word not in stop_words):
                if(word in word_positions[field]):
                    file = cache_list[field]
                    docs = getline(file,word_positions[field][word])[:-1]
                    docs = docs.split('|')
                    docs = [list(map(float,doc.split(':'))) for doc in docs]

                    if(not word_result): 
                        word_result = docs

                    else: 
                        word_result = mergePost(word_result,docs)
                        if(not word_result):
                            word_result = unionPost(word_result,docs)
        
        if(not result):
            result = word_result
        else:
            result = unionPost(result, word_result)
        
    return result


# In[10]:


def doomsday_lev1(fqueries, fields, cache_list = cache_list, word_positions = word_positions):
    
    fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}
    
    result = []
    best_result = []
    res_max_len = 0
    
    for idx,(fquery,field)in enumerate(zip(fqueries,fields)):
        
        result = mergeAll(fqueries[:idx] + fqueries[idx+1:], fields[:idx] + fields[idx+1:])    
        
        for f in fields_list:
            
            if(fields_list[f] == field):
                continue
            
            query_words = fquery.strip().split(' ')

            for word in query_words:

                word = stemmer.stemWord(word.lower())

                if(word not in stop_words):
                    if(word in word_positions[f]):
                        file = cache_list[f]
                        docs = getline(file,word_positions[f][word])[:-1]
                        docs = docs.split('|')
                        docs = [list(map(float,doc.split(':'))) for doc in docs]
                        
                        if(result): 
                            result = mergePost(result, docs)
                            if(result):
                                if(len(result) > res_max_len):
                                    best_result = result
                                    res_max_len = len(result)
                        
    return best_result


# In[11]:


def doomsday_lev2(fqueries, fields, cache_list = cache_list, word_positions = word_positions):
    
    fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}
    
    result = []
    best_result = []
    res_max_len = 0
    
    for idx1,(fquery1,field1) in enumerate(zip(fqueries,fields)):
        
        for idx2,(fquery2,field2) in enumerate(zip(fqueries,fields)):
            
            result = mergeAll(list(np.delete(np.array(fqueries),[idx1,idx2])), list(np.delete(np.array(fields),[idx1,idx2])))
            
            for f in fields_list:
            
                if(fields_list[f] == field1):
                    continue

                query_words = fquery1.strip().split(' ')

                for word in query_words:

                    word = stemmer.stemWord(word.lower())

                    if(word not in stop_words):
                        if(word in word_positions[f]):
                            file = cache_list[f]
                            docs = getline(file,word_positions[f][word])[:-1]
                            docs = docs.split('|')
                            docs = [list(map(float,doc.split(':'))) for doc in docs]

                            if(result): 
                                result = mergePost(result, docs)
                                if(result and len(result) > res_max_len):
                                    best_result = result
                                    res_max_len = len(result)

            if(result):
                for f in fields_list:
            
                    if(fields_list[f] == field2):
                        continue

                    query_words = fquery2.strip().split(' ')

                    for word in query_words:

                        word = stemmer.stemWord(word.lower())

                        if(word not in stop_words):
                            if(word in word_positions[f]):
                                file = cache_list[f]
                                docs = getline(file,word_positions[f][word])[:-1]
                                docs = docs.split('|')
                                docs = [list(map(float,doc.split(':'))) for doc in docs]

                                if(result): 
                                    result = mergePost(result, docs)
                                    if(result and len(result) > res_max_len):
                                        best_result = result
                                        res_max_len = len(result)
    return best_result


# In[12]:


def search(cache_list, queries, k = 5, word_positions = word_positions, title_list = title_list):
        
    fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}
        
    search_output = []
    
    for query in queries:
        
        flag = 0
        result = []
        
        if ':' in query:
            querio = re.sub(r'\w:',r'|',query).strip('|').split('|')
            fields = re.findall(r'\w:',query)
            fields = [f.split(':')[0] for f in fields]
            
            result = mergeAll(querio, fields)
            
            if(result and len(result) < k):
                result = unionPost(result, unionAll(querio, fields))

            if(not result):
                result = doomsday_lev1(querio, fields)
                if(not result):
                    result  = doomsday_lev2(querio, fields)
            
            if(not result):
                result = unionAll(querio, fields)
                            
        else:
            general_queries = query.split(' ')
            
            title_result = []
            body_result = []
            
            for gquery in general_queries:
                
                query_words = gquery.split(' ')

                field = 't'
                for word in query_words:
                    
                    word = stemmer.stemWord(word.lower())
                        
                    if(word not in stop_words):
                    
                        if(word in word_positions[field]):
                            file = cache_list[field]
                            docs = file.getline(word_positions[field][word])[:-1]
                            docs = docs.split('|')
                            docs = [list(map(float,doc.split(':'))) for doc in docs]

                            if(not title_result):
                                title_result = docs
                            else:
                                tmp = mergePost(result,docs)
                                if(not tmp):
                                    title_result = unionPost(title_result,docs)
                                else:
                                    title_result = tmp
                
                field = 'b'
                for word in query_words:
                    
                    word = stemmer.stemWord(word.lower())
                        
                    if(word not in stop_words):   
                        if(word in word_positions[field]):
                            file = cache_list[field]
                            docs = file.getline(word_positions[field][word])[:-1]
                            docs = docs.split('|')
                            docs = [list(map(float,doc.split(':'))) for doc in docs]

                            if(not body_result):
                                body_result = docs
                            else:
                                tmp = mergePost(body_result,docs)
                                if(not tmp):
                                    body_result = unionPost(body_result,docs)
                                else:
                                    body_result = tmp
            
            result = unionPost(title_result,body_result)
            
            
        if(result):
            result = sorted(result, key = lambda kv:(kv[1], kv[0]), reverse = True)
            doc_titles = []
            for idx,doc in enumerate(result):
                doc_titles.append(str(int(doc[0])) + ', ' + title_list[int(doc[0])-1])
#                 if(idx < k):
#                     print(doc[1])
                if(idx == k-1): break
            search_output.append(doc_titles)
        else:
            search_output.append(['No results found'])
        
    
    return search_output


# In[13]:


def give_output(query_file,out_file):
    with open(query_file,'r') as qf:
        with open(out_file,'w') as outf:
            for line in qf:
                k = line.split(',')[0]
                query = line.split(',')[1]
                start = time.perf_counter()
                results = search(cache_list,[query],int(k))
                end = time.perf_counter()
                total_time = end-start
                avg_time = total_time/int(k)
                for res in results[0]:
                    outf.write(res + '\n')
                outf.write(str(total_time) + ' ' + str(avg_time))
                outf.write('\n\n')


# In[14]:


query_file = './queries.txt'
out_file = './queries_op.txt'
give_output(query_file,out_file)


# In[ ]:




