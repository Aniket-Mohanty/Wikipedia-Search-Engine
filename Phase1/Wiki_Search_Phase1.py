
import os
import pickle
import linecache

import re
from collections import *
import time
import sys

import Stemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

index_path = sys.argv[1]
# query_path = sys.argv[2]
query_string = sys.argv[2]

stemmer = Stemmer.Stemmer('english')

file = open(index_path + 'word_positions.pickle','rb')
word_positions             =  pickle.load(file)
file.close()
file = open(index_path + 'title_position.pickle','rb')
title_list,title_position  =  pickle.load(file)
file.close()

def search(idx,index_path, queries, word_positions = word_positions, title_list = title_list, title_position = title_position):
        
    fields_list = ['t','b','c','i','r','e']
        
    search_output = []
    
    for query in queries:
        
        flag = 0
        result = []
        
        if ':' in query:
            querio = re.sub(r'\w:',r'|',query).strip('|').split('|')
            fields = re.findall(r'\w:',query)
            fields = [f.split(':')[0] for f in fields]
                        
            for field,quer in zip(fields,querio):
                
                flag2 = 0
                sub_result = []
                
                query_words = quer.strip().split(' ')
                
                for word in query_words:
                    
                    word = stemmer.stemWord(word.lower())

                    if(word not in stop_words):
                        if(word in word_positions[field]):
                            
                            word_posting = linecache.getline(os.path.join(index_path,field + '_1.txt'),word_positions[field][word])[:-1]
                            docs = [posting.split(':')[0] for posting in word_posting.split('|')]
                            
                            if(not flag2):
                                sub_result = list(set(sub_result) | set(docs))
                                flag2 = 1
                            else:
                                sub_result = list(set(sub_result) & set(docs))
                            
                            
                            if(not flag):
                                result = list(set(result) | set(docs))
                                flag = 1
                            else:
                                result = list(set(result) & set(docs))
                        else:
                            result = []
                print('\nQuery',idx,'Posting List of',quer,'for field',field,':',sub_result)
                            
        else:
            general_queries = query.split(' ')
            
            for gquery in general_queries:
                
                query_words = gquery.split(' ')
                
                for word in query_words:
                    
                    word = stemmer.stemWord(word.lower())
                    
#                     for field in fields_list:
                    field = 'b'
                    if(word not in stop_words):
                        if(word in word_positions[field]):

                            word_posting = linecache.getline(index_path + field + '_1.txt',word_positions[field][word])[:-1]
                            docs = [posting.split(':')[0] for posting in word_posting.split('|')]

                            if(not flag):
                                result = list(set(result) | set(docs))
                                flag = 1
                            else:
                                result = list(set(result) & set(docs))
                                
            print('\nQuery',idx,'Posting List of',query,':',result)
            
            
        if(result):
            doc_titles = []
            for doc in result:
                doc_titles.append(title_list[int(doc)-1])
            search_output.append(doc_titles)
        else:
            search_output.append(['No results found'])
        
    
    return search_output

queries = query_string.split('\\n')
print(queries)
for idx,query in zip(range(len(queries)),queries):
    search(idx,index_path,[query])
    
    
# with open(query_path,'r') as f:
#     idx = 0
#     for query in f:
#         search(idx,index_path,[query])
#         idx+=1

# start = time.perf_counter()
# result = search(index_path,['and'])
# end = time.perf_counter()
# print(end - start)

