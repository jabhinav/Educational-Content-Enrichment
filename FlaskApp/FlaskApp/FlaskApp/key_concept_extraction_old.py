from __future__ import division
import nltk
import re
import string as string_lib
import pandas as pd
import numpy
import networkx as nx
import math
import wikipedia
import itertools
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

## Loaction of all jar files
stanford_pos_dir = '/Users/rishimadhok/Desktop/Abhinav_IBM/stanford-postagger-full-2017-06-09/'
# /Users/rishimadhok/Desktop/Abhinav_IBM/stanford-postagger-full-2017-06-09/
# /home/abhinav/NLP/softwares/stanford-postagger-full-2017-06-09/
eng_model_filename_pos = stanford_pos_dir + 'models/english-left3words-distsim.tagger'
my_path_to_pos_jar= stanford_pos_dir + 'stanford-postagger.jar'


## Global Variables
p = re.compile('\([^()]*\)')                             # won't work in cases where bracket in bracket
reg_punc = re.compile('[%s]' % re.escape(string_lib.punctuation))
stop = stopwords.words('english')
concept_count = {}
pruned_concepts = {}
concept_wiki_article = {}
concept_sentences = {}
final_wiki_concepts =[]
wordnet_lemmatizer = WordNetLemmatizer()

## Global Scope Functions
tokenize = lambda doc: doc.lower().split(" ")

def remove_IOBtags(IOB_tagged):
    string =""
    iterator = iter(IOB_tagged)
    concepts = []

    for item in IOB_tagged:
        if item[2] == 'B-NP':
            if string!="":
                concept = ' '.join(string.split())
                concepts.append(concept)
                if string not in concept_count:
                    concept_count[string] = 1
                else:
                    concept_count[string] += 1

                string = ""
            if item[0].lower() not in stop:                                                     # removes chunked entities which start with stop words (if any)
                to_regex = wordnet_lemmatizer.lemmatize(item[0])
                regexed = p.sub('',to_regex)
                no_punc = reg_punc.sub('',regexed)                                              # removes punctuations
                result = ''.join(i for i in no_punc if not i.isdigit())                         # removes numbers
                result = ' '.join(result.split())
                string = result

        elif item[2] == 'I-NP':
            to_regex = wordnet_lemmatizer.lemmatize(item[0])
            regexed = p.sub('',to_regex)
            no_punc = reg_punc.sub('', regexed)
            result = ''.join(i for i in no_punc if not i.isdigit())
            result = ' '.join(result.split())
            string = string + " " + result

    if string != "":
        concept = ' '.join(string.split())
        concepts.append(concept)
        if string not in concept_count:
            concept_count[string] = 1
        else:
            concept_count[string] += 1



def parsing(sent):

    parser = StanfordDependencyParser(path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_parser_jar)
    result = parser.raw_parse(sent)
    dep = next(result)
    parsed = list(dep.triples())
    return parsed

## Function does IOB Chunking based on regex rules(grammar)
def chunking(sent,grammar):
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(sent, trace=0)
    # tree.draw()
    IOB_tagged = nltk.tree2conlltags(tree)
    return IOB_tagged

## Function ignores common words based on their occurence(count) in BBC Corpus
def prune_concepts_WordLevel():
    freq_threshold = 50                                         # Words with frequency greater than the threshold are ignored
    df = pd.read_csv("/Users/rishimadhok/Desktop/Abhinav_IBM/BBCData.csv", header=None)
    for key in concept_count:
        word = wordnet_lemmatizer.lemmatize(key)                # Lemmatize each word identified
        test = df[df[0]== word]
        if not test.empty:
            row_num = df.index[df[0]== word].item()
            freq = test[1][row_num]
            # print("Word: %s, Freq: %s" % (word,freq))
            if (freq < freq_threshold):
                pruned_concepts[key] = concept_count[key]
        else:
            pruned_concepts[key] = concept_count[key]

## for jaccard and tf-idf python implementation -> http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def Wikipedia_aritcle_matching():
    for key in pruned_concepts:
        wiki_articles = wikipedia.search(key)
        concept_wiki_article[key] = []
        # print("\nConcept (%s) has following wikipedia articles:" %(key))
        # print(wiki_articles)
        if len(wiki_articles) != 0:
            tokenized_key = ' '.join(tokenize(key)).split()      # join-split combo removes empty strings that might appear in tokenised list (this tokenised list already has lemmatized words which were taken care of in function remove_IOBtags())
            regexed_wiki_articles = [p.sub('',d.lower()) for d in wiki_articles]
            no_punc_wiki_articles = [reg_punc.sub('', d) for d in regexed_wiki_articles]

            ## Read every wiki article, then tokenise it, output list might contain empty string (while removing brackets enclosed text) so remove them, lemmatise every word
            tokenized_wiki_aritcles = [[wordnet_lemmatizer.lemmatize(word) for word in ' '.join(tokenize(d)).split()] for d in no_punc_wiki_articles]
            # all_tokens_set = set([item for sublist in tokenized_wiki_aritcles for item in sublist])

            for i, tokenized_article in enumerate(tokenized_wiki_aritcles):
                score = jaccard_similarity(tokenized_key,tokenized_article)
                article = wiki_articles[i]
                if score == 1:                                                            # Use other means to compute similarity between concept extracted and their corresponding wikipedia articles (for time being jaccard similarity used)
                    concept_wiki_article[key].append(article)                                 # Concepts with exact matching are stored only
                    if article not in final_wiki_concepts:
                        final_wiki_concepts.append(article)



                elif len(set(tokenized_article)-set(tokenized_key))==0:                  # if some wiki article is present as a substring in extracted concept, then also add it as a valid concept but do this only after verifying that exact match article dose not exist
                    concept_wiki_article[key].append(article)
                    if article not in final_wiki_concepts:
                        final_wiki_concepts.append(article)


            if len(concept_wiki_article[key])==0:
                concept_wiki_article[key].append('')

        else:
            concept_wiki_article[key].append('')
        # print("Score:%d"%(max))


def main():

    with open ('fetched_url_download', 'rb') as fp:
        text = pickle.load(fp) 
    # with open('/home/abhinav/PycharmProjects/video_enrichment/text.txt', 'r') as myfile:
    #     text = myfile.read().replace('\n', '')

    # text = """Natural language processing (NLP) is a field of computer science, artificial intelligence and computational linguistics concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language corpora."""
    # text = text
    # text  = "Concepts present in text are outline of machine learning, data mining, statistics, cluster analysis, algorithms like logic, pseudo code."
    # text = p.sub('',text)
    sentences = nltk.sent_tokenize(text)

    for sent in sentences:
        sentence = sent.lower()                     # Lower Case the whole sentence
        sentence = p.sub('',sentence)                   # Removing anything enclosed within brackets
        print(sentence)

        ## TAGGING
        st_tag = StanfordPOSTagger(model_filename=eng_model_filename_pos, path_to_jar=my_path_to_pos_jar)
        tagged_sentence = st_tag.tag(word_tokenize(sentence))
        # print(tagged_sentence)

        ## ENTITY RECOGNITION
        # st_ner = StanfordNERTagger(model_filename=eng_model_filename_ner, path_to_jar=my_path_to_ner_jar)
        # print(st_ner.tag('Rami Eid is studying at Stony Brook University in NY'.split()))

        ## PARSING
        # print(parsing(sentence))

        ## Chunking Using Regex
        regex_exps = ["NP: {<JJ|NN.?>*<NN.?>}", "NP: {<JJ|NN.?>*<NN.?><IN>?<JJ|NN.?>*<NN.?>}", "NP: {<JJ>*<NN.?>+}"] # Include the following pattern to count conjuctions "NP: {<JJ|NN.?>*<NN.?><CC>?<JJ|NN.?>*<NN.?>}"
        for grammar in regex_exps:
            IOB_tagged = chunking(tagged_sentence,grammar)
            remove_IOBtags(IOB_tagged)

    # print(concept_count)

    ## Prune concepts on word level using word frequency count on BBC corpus
    prune_concepts_WordLevel()
    print("Pruned concepts are:",pruned_concepts)

    ## Identify Wikipedia articles(titles) that match concepts extracted from the text if Jaccard Similarity is one or if wikipedia title is a part of concept extracted
    Wikipedia_aritcle_matching()
    print("\n",concept_wiki_article)
    print("\nFinal List Of Concepts:",final_wiki_concepts)

    with open('final_concepts', 'wb') as fp:
            pickle.dump(final_wiki_concepts, fp)

if __name__ == "__main__":
    main()
