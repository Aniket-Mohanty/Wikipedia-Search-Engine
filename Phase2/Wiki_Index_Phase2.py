#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pickle
import sys
import gc
from contextlib import ExitStack

import xml.etree.cElementTree as et
import re
import heapq

from collections import *
from math import *
import numpy as np
import time

import Stemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

start_time = time.perf_counter()
stop_words = set(stopwords.words('english'))

stop_dict = defaultdict(int)
for word in stop_words:
    stop_dict[word] = 1


# In[6]:


# wiki_xml = "../WikiDump_1/WikiDump_1.xml-p1p30303"
index_path = './index_data/'
index_full_path = './full_index/'
final_index = './final_index/'

if(not os.path.isdir(index_path)): os.mkdir(index_path)
if(not os.path.isdir(index_full_path)): os.mkdir(index_full_path)
if(not os.path.isdir(final_index)): os.mkdir(final_index)


# In[7]:


# num_xml = 5
data_path = '../Phase2_Data/'
wiki_xml_list = os.listdir(data_path)


# In[8]:


# wiki_xml_list


# In[9]:


def get_external_links(body):
    external_links = []
    lines = body.split("==")[-1]
    lines = lines.split("\n")
 
    for line in lines:
        if re.match(r"\*(.*)", line):
            external_links.append(line)
 
    return external_links


# In[10]:


def clean_wiki(body):
    
    body = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ',body,flags = re.DOTALL)
    body = re.sub('<!--.*?-->',' ',body,flags = re.DOTALL)
    body = re.sub('<math([> ].*?)(</math>|/>)',' ',body,flags = re.DOTALL)
    body = re.sub(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])',' ',body,flags = re.DOTALL)
    body = re.sub(r'{{v?cite(.*?)}}',' ',body,flags = re.DOTALL)

    References = re.findall("<ref>(.*?)</ref>", body)
    Infobox = re.findall(r"\{\{Infobox (.*?)\}\}", body, flags = re.DOTALL)
    Category = re.findall(r"\[\[Category:(.*?)\]\]", body)
    External = get_external_links(body)

    body = re.sub('<.*?>',' ',body,flags = re.DOTALL)
    body = re.sub('{{([^}{]*)}}','',body,flags = re.DOTALL)
    body = re.sub('{{([^}]*)}}','',body,flags = re.DOTALL)

    return body.lower(), Infobox, Category, References, External


# In[11]:


stemmer = Stemmer.Stemmer('english')


# In[12]:


title_position = []
title_list = []


# In[13]:


start = time.perf_counter()
PageCount = 0
title_num = 0
MAX_FILES = 30000
file_count = 0

for wiki_xml in wiki_xml_list:
    
    title_index = defaultdict(list)
    body_index = defaultdict(list)
    category_index = defaultdict(list)
    infobox_index = defaultdict(list)
    reference_index = defaultdict(list)
    external_index = defaultdict(list)
    
    stem_dict = defaultdict(int)
    
#     _ = gc.collect()
    
    wiki_xml = os.path.join(data_path,wiki_xml)
    
    for idx, (event, elem) in enumerate(et.iterparse(wiki_xml, events=('start', 'end'))):

        tag = elem.tag.split('}')[-1]

        if event == 'start':
            if tag == 'page':
                title_dict = defaultdict(int)
                infobox_dict = defaultdict(int)
                category_dict = defaultdict(int)
                reference_dict = defaultdict(int)
                external_dict = defaultdict(int)
                id = -1
                redirect = ''
                inrevision = False
                ns = 0
                body_dict = defaultdict(int)
                body_empty = False
            elif tag == 'revision':
                # Do not pick up on revision id's
                inrevision = True

        else:
            if tag == 'title':
                title = elem.text
                if(title):
                    title_list.append(title)
                    title_num += 1
            elif tag == 'id' and not inrevision:
                id = int(elem.text)
            elif tag == 'redirect':
                redirect = elem.attrib['title']
            elif tag == 'ns':
                ns = int(elem.text)
            elif tag == 'text':
                body = elem.text
                if(body == None):
                    body_empty = True
            elif tag == 'page':
                PageCount += 1

                if(not body_empty):
                    body,Infobox,Category,Reference,External = clean_wiki(body)
                
#################### Acquiring word counts per document ####################

                #### Title ####
                title_words = re.split("[^a-zA-Z0-9]",title.lower())
                for word in title_words:
                    if(word):
                        if(not stem_dict[word]):
                            stem_dict[word] = stemmer.stemWord(word)
                        word = stem_dict[word]
                        if(len(word) > 2):
                            title_dict[word] += 1

                if(not body_empty):
                #### Infobox ####
                    for info in Infobox:
                        info_words = re.split("[^a-zA-Z0-9]",info.lower())
                        for word in info_words:
                            if(word and not stop_dict[word]):
                                if(not stem_dict[word]):
                                    stem_dict[word] = stemmer.stemWord(word)
                                word = stem_dict[word]
                                if(len(word) > 2):
                                    infobox_dict[word] += 1

                #### Category ####
                    for cate in Category:
                        cate_words = re.split("[^a-zA-Z0-9]",cate.lower())
                        for word in cate_words:
                            if(word and not stop_dict[word]):
                                if(not stem_dict[word]):
                                    stem_dict[word] = stemmer.stemWord(word)
                                word = stem_dict[word]
                                if(len(word) > 2):
                                    category_dict[word] += 1

                #### Reference ####
                    for ref in Reference:
                        ref_words = re.split("[^a-zA-Z0-9]",ref.lower())
                        for word in ref_words:
                            if(word and not stop_dict[word]):
                                word = stemmer.stemWord(word)
                                if(len(word) > 2):
                                    reference_dict[word] += 1

                #### External ####
                    for ext in External:
                        ext_words = re.split("[^a-zA-Z0-9]",ext.lower())
                        for word in ext_words:
                            if(word and not stop_dict[word]):
                                if(not stem_dict[word]):
                                    stem_dict[word] = stemmer.stemWord(word)
                                word = stem_dict[word]
                                if(len(word) > 2):
                                    external_dict[word] += 1


                ##### Body #####
                    body_words = re.split("[^a-zA-Z0-9]",body)
                    for word in body_words:
                        if(word and not stop_dict[word]):
                            if(not stem_dict[word]):
                                stem_dict[word] = stemmer.stemWord(word)
                            word = stem_dict[word]
                            if(len(word) > 2):
                                body_dict[word] += 1

####################### Index Creation #######################

                for word in title_dict :
                    tf = round(1 + log10(title_dict[word]),3)
                    title_index[word].append(':'.join((str(PageCount),str(tf))))

                if(not body_empty):
                
                    for word in body_dict :
                        tf = round(1 + log10(body_dict[word]),3)
                        body_index[word].append(':'.join((str(PageCount),str(tf))))

                    for word in category_dict :
                        tf = round(1 + log10(category_dict[word]),3)
                        category_index[word].append(':'.join((str(PageCount),str(tf))))

                    for word in infobox_dict :
                        tf = round(1 + log10(infobox_dict[word]),3)
                        infobox_index[word].append(':'.join((str(PageCount),str(tf))))

                    for word in reference_dict :
                        tf = round(1 + log10(reference_dict[word]),3)
                        reference_index[word].append(':'.join((str(PageCount),str(tf))))

                    for word in external_dict :
                        tf = round(1 + log10(external_dict[word]),3)
                        external_index[word].append(':'.join((str(PageCount),str(tf))))


############## File creation per MAX_FILES pages ##############

                if(PageCount % MAX_FILES == 0):

                    stem_dict = defaultdict(int)

                    body_index = OrderedDict(sorted(body_index.items()))
                    title_index = OrderedDict(sorted(title_index.items()))
                    category_index = OrderedDict(sorted(category_index.items()))
                    infobox_index = OrderedDict(sorted(infobox_index.items()))
                    reference_index = OrderedDict(sorted(reference_index.items()))
                    external_index = OrderedDict(sorted(external_index.items()))

                    file_name = index_path + 't' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in title_index:
                        posting = '|'.join(title_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    file_name = index_path + 'b' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in body_index:
                        posting = '|'.join(body_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    file_name = index_path + 'c' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in category_index:
                        posting = '|'.join(category_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    file_name = index_path + 'i' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in infobox_index:
                        posting = '|'.join(infobox_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    file_name = index_path + 'r' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in reference_index:
                        posting = '|'.join(reference_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    file_name = index_path + 'e' + '_' + str(file_count) + '.txt'
                    file = open(file_name, 'w+')
                    for word in external_index:
                        posting = '|'.join(external_index[word]) + '\n'
                        posting = word + '>' + posting
                        file.write(posting)
                    file.close()

                    title_index = defaultdict(list)
                    body_index = defaultdict(list)
                    category_index = defaultdict(list)
                    infobox_index = defaultdict(list)
                    reference_index = defaultdict(list)
                    external_index = defaultdict(list)

#                     _ = gc.collect()

                    file_count += 1

            elem.clear()
    #     if(PageCount == 30):
    #         break

########### File creation for last batch of pages ###########
    
    stem_dict.clear()

    body_index = OrderedDict(sorted(body_index.items()))
    title_index = OrderedDict(sorted(title_index.items()))
    category_index = OrderedDict(sorted(category_index.items()))
    infobox_index = OrderedDict(sorted(infobox_index.items()))
    reference_index = OrderedDict(sorted(reference_index.items()))
    external_index = OrderedDict(sorted(external_index.items()))

    file_name = index_path + 't' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in title_index:
        posting = '|'.join(title_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

    file_name = index_path + 'b' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in body_index:
        posting = '|'.join(body_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

    file_name = index_path + 'c' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in category_index:
        posting = '|'.join(category_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

    file_name = index_path + 'i' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in infobox_index:
        posting = '|'.join(infobox_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

    file_name = index_path + 'r' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in reference_index:
        posting = '|'.join(reference_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

    file_name = index_path + 'e' + '_' + str(file_count) + '.txt'
    file = open(file_name, 'w+')
    for word in external_index:
        posting = '|'.join(external_index[word]) + '\n'
        posting = word + '>' + posting
        file.write(posting)
    file.close()

#     _ = gc.collect()
    
    file_count += 1

end = time.perf_counter()
diff1 = end-start 
print(diff1)


# In[14]:


if(len(title_list) != title_num):
    print('Warning: Might need to track title positions')


# In[15]:


start = time.perf_counter()
word_position = defaultdict(dict)
fields_list = {'t':'title','b':'body','c':'category','i':'infobox','r':'reference','e':'external'}
token_count = 0
for field in fields_list.keys():
    files = []
#     _ = gc.collect()
    prev_word = ''
    LineNum = 1
    with ExitStack() as stack:
        for idx in range(file_count):
            file_name = index_path + field + '_' + str(idx) + '.txt'
            files.append(stack.enter_context(open(file_name)))
        with open(index_full_path + fields_list[field] + '.txt', 'w') as f:
                file_iter = heapq.merge(*files)
                for idx,line in enumerate(file_iter):
                    line = line.split('>')
                    word = line[0]
                    posting_list = line[1][:-1]
                    if(word == prev_word):
                        f.write('|' + posting_list)
                    else:
                        word_position[field][word], LineNum = LineNum, LineNum+1
                        if(idx == 0):
#                             f.write(word + '-' + posting_list)
                            f.write(posting_list)
                        else:
#                             f.write('\n' + word + '-' + posting_list)
                            f.write('\n' + posting_list)
                        
                        if(field == 'b'): token_count += 1
                
                    prev_word = word
end = time.perf_counter()
print(end-start)


# In[17]:


print('Number of tokens: ',token_count)


# In[ ]:


start = time.perf_counter()
term_weights = {'title':5,'body':2,'category':2,'infobox':3,'reference':1,'external':1}
for field in fields_list.values():
    with open(index_full_path + field + '.txt','r') as fr:
        with open(final_index + field + '.txt','w+') as fw:
            for line in fr:
                docs = line.split('|')
                docs = [doc.split(':') for doc in docs]
                docs = dict(sorted(docs,key = lambda kv:(int(kv[0]), kv[1])))

                values = np.array(list(map(float,docs.values())))
                num_docs_with_word = len(docs)
                idf = log10(PageCount/num_docs_with_word)
                values = np.round(term_weights[field]*values*idf,2)

                new_line = []
#                 _ = gc.collect()
                
                for doc,val in zip(docs.keys(),values):
                    new_line.append(doc + ':' + str(val))

                new_line = '|'.join(new_line)

                fw.write(new_line + '\n')
end = time.perf_counter()
print(end - start)


# In[ ]:


start = time.perf_counter()

word_file = open(final_index + "word_positions.pickle", "wb+")
pickle.dump(word_position, word_file)
word_file.close()

title_file = open(final_index + "title_position.pickle", "wb+")
pickle.dump(title_list,title_file)
title_file.close()

end = time.perf_counter()
diff4 = end-start
print(diff4)


# In[ ]:


print('Total time: ', end-start_time)


# In[ ]:





# In[ ]:




