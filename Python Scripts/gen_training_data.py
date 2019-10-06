# -*- coding: utf-8 -*-
"""
This prepocesses the dataset to produce a csv file with training labels. Following pre-precessing steps are employed:
1. Remove non-ASCII characters
2. Remove bracket enclosed text
3. Retain square bracket enclosed text
"""
import os
import csv
import re
from nltk import word_tokenize

def remove_non_ascii(text):
    s = ''.join([i if ord(i) < 128 else '' for i in text])
#    return s.replace('\n','').replace('\r','').replace('\t','');
    return ' '.join(s.split())
    
def sentence_splitter(text):
    # t = remove_question_number(text)
    s = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return s
    
def remove_question_number(q):
    re_list = [re.compile(r'Q\.*\s*No\.*\s*[0-9]+'),
               re.compile(r'Q\.*\s*[0-9]+\s*\.*'),                
                re.compile(r'[0-9]+\s*\.+')]
    
    for r in re_list:
        q = re.sub(r,'',q)
    
    return q
    
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def read_csv(csvpath):
    csv_lines = []
    print(csvpath)
    with open(csvpath, 'r') as c:
        reader = csv.reader(c)
        for row in reader:           
#            csv_lines.append(map(cleanhtml,row));
            csv_lines.append([cleanhtml(row[0]), cleanhtml(row[1]), row[2]])
                
    return csv_lines

def write_csv(data, fname):
    with open(os.path.join(folderpath, fname), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)


def main():
    pos = []
    p = re.compile('\([^()]*\)')                        ## for removing outermost brackets (in case of bracket in bracket)

    with open(os.path.join(folderpath,'samples_test_positive.txt'),'r') as f:
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            if 10<len(line):                      # To remove sentences with single word or very few words or which are empty
                processed_1 = p.sub('', line)           # Removes bracket enclosed text
                # processed_1 = p.sub('',processed_1)     # For bracket in bracket (upto levels)
                processed_2 = remove_non_ascii(processed_1)
                processed_3 = processed_2.replace('[','').replace(']','').replace(';',',')  # Remove [,] and replace ';' with ','. NOTE: This step is highly dataset specific. Semicolon is replaced because reading it was putting parts of sentence separated by ; into separate columns of csv
                pos.append([processed_3])

    write_csv(pos, 'samples_test_positive.csv')


folderpath = '/home/abhinav/PycharmProjects/video_enrichment/dataset/Application_Extraction'

if __name__=="__main__":
    # main()


    file = os.path.join(folderpath,"original_applications_sec_list.txt")
    with open(file,'r') as f:
        sentences = f.readlines()
    file_2 = os.path.join(folderpath,"proto_negative.txt")
    # file_3 = os.path.join(folderpath,"")
    g = open(file_2,'w')
    print("Num Sentences:%i"%len(sentences))
    for i, sent in enumerate(sentences):
        if i<1500 and i >999:
            sent = sent.lower()
            lis = word_tokenize(sent)
            if "used" in lis or "uses" in lis:
                continue
            elif "applied"in lis:
                continue
            elif "applications" in lis or "application" in lis:
                continue
            elif "utilized" in lis or "useful" in lis:
                continue
            elif "examples" in lis:
                continue
            elif "implemented" in lis:
                continue
            else:
                print(sent)
                value = input("Enter zero or one for sent num %i:"%i)
                if value == '1':
                    g.write("%s" % sent)

    g.close()