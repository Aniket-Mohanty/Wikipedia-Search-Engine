
import os
import pickle
import sys

import xml.etree.cElementTree as et
import re

from collections import *
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

wiki_xml = sys.argv[1]
index_path = sys.argv[2]
stat_path = sys.argv[3]

def get_external_links(body):
    external_links = []
    lines = body.split("==")[-1]
    lines = lines.split("\n")
 
    for line in lines:
        if re.match(r"\*(.*)", line):
            external_links.append(line)
 
    return external_links

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

stemmer = Stemmer.Stemmer('english')

title_position = []
title_list = []

title_index = defaultdict(list)
body_index = defaultdict(list)
category_index = defaultdict(list)
infobox_index = defaultdict(list)
reference_index = defaultdict(list)
external_index = defaultdict(list)

stem_dict = defaultdict(int)
token_count = 0
token_clean_count = defaultdict(int)

start = time.perf_counter()
PageCount = 0
title_num = 0

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
        elif tag == 'page':
            PageCount += 1
            
            body,Infobox,Category,Reference,External = clean_wiki(body)
            ############# Acquiring word counts per document #############
            
            #### Title ####
            title_words = re.split("[^a-zA-Z0-9]",title.lower())
            for word in title_words:
                if(word):
                    token_count += 1
                    word = stemmer.stemWord(word)
                    if(len(word) > 2):
                        token_clean_count[word] = 1
                        title_dict[word] += 1
                        
            #### Infobox ####
            for info in Infobox:
                info_words = re.split("[^a-zA-Z0-9]",info.lower())
                for word in info_words:
                    token_count += 1
                    if(word and not stop_dict[word]):
                        if(not stem_dict[word]):
                            stem_dict[word] = stemmer.stemWord(word)
                        word = stem_dict[word]
                        if(len(word) > 2):
                            token_clean_count[word] = 1
                            infobox_dict[word] += 1  
                        
            #### Category ####
            for cate in Category:
                cate_words = re.split("[^a-zA-Z0-9]",cate.lower())
                for word in cate_words:
                    token_count += 1
                    if(word and not stop_dict[word]):
                        if(not stem_dict[word]):
                            stem_dict[word] = stemmer.stemWord(word)
                        word = stem_dict[word]
                        if(len(word) > 2):
                            token_clean_count[word] = 1
                            category_dict[word] += 1
            
            #### Reference ####
            for ref in Reference:
                ref_words = re.split("[^a-zA-Z0-9]",ref.lower())
                for word in ref_words:
                    token_count += 1
                    if(word and not stop_dict[word]):
                        word = stemmer.stemWord(word)
                        if(len(word) > 2):
                            token_clean_count[word] = 1
                            reference_dict[word] += 1
            
            #### External ####
            for ext in External:
                ext_words = re.split("[^a-zA-Z0-9]",ext.lower())
                for word in ext_words:
                    token_count += 1
                    if(word and not stop_dict[word]):
                        if(not stem_dict[word]):
                            stem_dict[word] = stemmer.stemWord(word)
                        word = stem_dict[word]
                        if(len(word) > 2):
                            token_clean_count[word] = 1
                            external_dict[word] += 1
            
            
            #### Body ####
            body_words = re.split("[^a-zA-Z0-9]",body)
            for word in body_words:
                token_count += 1
                if(word and not stop_dict[word]):
                    if(not stem_dict[word]):
                        stem_dict[word] = stemmer.stemWord(word)
                    word = stem_dict[word]
                    if(len(word) > 2):
                        token_clean_count[word] = 1
                        body_dict[word] += 1
            
            ################ Index Creation ################
            
            for word in body_dict :     body_index[word].append(':'.join((str(PageCount),str(body_dict[word]))))

            for word in title_dict :    title_index[word].append(':'.join((str(PageCount),str(title_dict[word]))))

            for word in category_dict : category_index[word].append(':'.join((str(PageCount),str(category_dict[word]))))

            for word in infobox_dict :  infobox_index[word].append(':'.join((str(PageCount),str(infobox_dict[word]))))
            
            for word in reference_dict :  reference_index[word].append(':'.join((str(PageCount),str(reference_dict[word]))))

            for word in external_dict :  external_index[word].append(':'.join((str(PageCount),str(external_dict[word]))))

            
        elem.clear()
#     if(PageCount == 30):
#         break

        
end = time.perf_counter()
diff1 = end-start 
# print(diff1)

start = time.perf_counter()
######## Sorting of indexes ########

body_index = OrderedDict(sorted(body_index.items()))
title_index = OrderedDict(sorted(title_index.items()))
category_index = OrderedDict(sorted(category_index.items()))
infobox_index = OrderedDict(sorted(infobox_index.items()))
reference_index = OrderedDict(sorted(reference_index.items()))
reference_index = OrderedDict(sorted(reference_index.items()))
external_index = OrderedDict(sorted(external_index.items()))
end = time.perf_counter()
diff2 = end-start
# print(diff2)

start = time.perf_counter()

word_position = defaultdict(dict)
fields_list = {'t':title_index,'b':body_index,'c':category_index,'i':infobox_index,'r':reference_index,'e':external_index}

for field in fields_list.keys():
    LineNum = 1
    file_name = os.path.join(index_path , field + '_1.txt')
    file = open(file_name, 'w+')
    for word in fields_list[field]:
        posting = '|'.join(fields_list[field][word]) + '\n'
        file.write(posting)
        word_position[field][word], LineNum = LineNum, LineNum+1
    file.close()
    
end = time.perf_counter()
diff3 = end-start
# print(diff3)

start = time.perf_counter()

word_file = open(os.path.join(index_path,"word_positions.pickle"), "wb+")
pickle.dump(word_position, word_file)
word_file.close()

title_file = open(os.path.join(index_path + "title_position.pickle"), "wb+")
pickle.dump((title_list,title_position),title_file)
title_file.close()

end = time.perf_counter()
diff4 = end-start
# print(diff4)

Total_time = end - start_time

with open(stat_path,'w+') as f:
    f.write(str(token_count) + '\n')
    f.write(str(len(token_clean_count)))

# line_count_total = 0
# for field in ['t','b','c','i','r','e']:
#     lines = 0
#     with open(index_path + field + '_1.txt') as f:
#         for line in f: lines+=1
#     line_count_total += lines

# print(Total_time,token_count,len(token_clean_count))
