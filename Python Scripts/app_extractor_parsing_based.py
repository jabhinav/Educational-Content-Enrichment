"""
This python script parses the dump of latest wikipedia articles created by wikiextractor. wikiextractor converted every wiki page into a dictionary. Following code tokenizes the complete text
present in the wiki page into sentences. "Applications" section is identified as separate sentence ("Applications.") in the tokenized list. The sentence which is an application of a concept
is assummed to be the immediate or next to next sentence of the "Applications." sentence.
"""

import json
import os
import nltk

location ="/home/abhinavj/wikiextractor/dump/"

sent1 = []
sent2 = []
num_found =0
sub_directories = os.listdir(location)
for x in sub_directories:
    sub_dir = os.path.join(location,x)
    files = os.listdir(sub_dir)
    for file in files:
        Data = []
        with open(os.path.join(sub_dir,file),'r') as inputData:
            for line in inputData:
                try:
                   Data.append(json.loads(line.rstrip(';\n')))
                except ValueError:
                    print ("Skipping invalid line {0}".format(repr(line)))

            for dic in Data:
                print("Title: ",dic['title'])
                text = dic['text']
                sentences = nltk.sent_tokenize(text)
                app_string = "Applications."
                if app_string in sentences:
                    num_found+=1
                    position = sentences.index(app_string)
                    if position != len(sentences)-1:                # To avoid index error
                        sent1.append(sentences[position+1])
                    if position != len(sentences)-2:
                        sent2.append(sentences[position+2])


print("Number of applications found : %i"%num_found)

list1 = open(os.path.join(location,'main_list.txt'),'w')
for item in sent1:
    list1.write("%s\n" % item)
list1.close()

list2 = open(os.path.join(location,'sec_list.txt'),'w')
for item in sent2:
    list2.write("%s\n" % item)
list2.close()