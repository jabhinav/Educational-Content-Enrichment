"""
Do before release:
1. Corref resolution

Caveats:
1. If a concept has a Wiki article in it, then also it won't be considered as a key-concept : We have only considered exact matches
2. If a sentence has multiple subjects, then we'll have many-one matching of sentences to concepts
3. If subject of the sentence appears multiple times in the sentence (in different phrases), then subject containing phrase appearing first will be assigned the sentence. (To resolve this, use relations like compound, amod and nn)
4. Since, we are ignoring bracket enclosed text, this also leads to multiple matches in exact matching of wiki-articles to concept
"""

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
from py2neo import Graph, authenticate

## Loaction of all jar files
stanford_pos_dir = '/home/abhinav/NLP/softwares/stanford-postagger-full-2017-06-09/'
eng_model_filename_pos = stanford_pos_dir + 'models/english-left3words-distsim.tagger'
my_path_to_pos_jar= stanford_pos_dir + 'stanford-postagger.jar'

# stanford_ner_dir = '/home/abhinav/NLP/softwares/stanford-ner-2017-06-09/'
# eng_model_filename_ner = stanford_ner_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'
# my_path_to_ner_jar = stanford_ner_dir + 'stanford-ner.jar'

stanford_parser_dir = '/home/abhinav/NLP/softwares/stanford-parser-full-2017-06-09/'
eng_model_filename_parser = stanford_parser_dir + "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz"
my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.8.0-models.jar"
my_path_to_parser_jar = stanford_parser_dir + "stanford-parser.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=my_path_to_parser_jar,path_to_models_jar=my_path_to_models_jar)

## Global Variables
p = re.compile('\([^()]*\)')                             # won't work in cases where bracket in bracket
reg_punc = re.compile('[%s]' % re.escape(string_lib.punctuation))
stop = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

## Global Scope Functions
tokenize = lambda doc: doc.lower().split(" ")


def process_word(word):
    to_regex = wordnet_lemmatizer.lemmatize(word)
    regexed = p.sub('', to_regex)
    no_punc = reg_punc.sub('', regexed)  # removes punctuations
    result = ''.join(i for i in no_punc if not i.isdigit())  # removes numbers
    result = ' '.join(result.split())
    return result

def get_subject(sent):
    noun_subjects = []
    result = dependency_parser.raw_parse(sent)
    dep = next(result)
    dependency_parsed = list(dep.triples())

    for_subject = ['nsubj','nsubjpass']
    for item in dependency_parsed:
        head = item[0][0]
        head_tag = item[0][1]
        relation = item[1]
        dependent = item[2][0]
        dependent_tag = item[2][1]

        if relation in for_subject:
            if dependent_tag.startswith('N'):
                result = process_word(dependent)
                if result not in noun_subjects:
                    noun_subjects.append(result)

    print("Subjects:",noun_subjects)
    return noun_subjects

def remove_IOBtags(IOB_tagged,subjects,sent,concept_count,concept_sentences):
    string = ""
    iterator = iter(IOB_tagged)
    concepts = []
    num_times = len(subjects) # Number of times the same sentence has to be mapped to a particular subject(concept) = Total Number of subjects identified in the Sentence. . If subjects list is empty, then no sentence will be mapped.
    map_sent_string = False


    for item in IOB_tagged:
        if item[2] == 'B-NP':
            if string!="":
                concept = ' '.join(string.split())
                concepts.append(concept)
                if string not in concept_count:
                    concept_count[string] = 1
                else:
                    concept_count[string] += 1

                if num_times > 0 and map_sent_string == True:       # If word which is the subject of the sentence is in the string and also sentence has not been mapped to it, map the sentence

                    if string not in concept_sentences:
                        concept_sentences[string] = [sent]
                    else:
                        if sent not in concept_sentences[string]:   # So that you dont add same sentence multiple times to the same concept. Note that same sentence can be added multiple times but to different strings in case we have multiple subjects
                            concept_sentences[string].append(sent)

                    num_times = num_times - 1
                    map_sent_string = False

                string = ""

            if item[0].lower() not in stop:    # removes chunked entities which start with stop words (if any)
                result = process_word(item[0])
                string = result

                if result in subjects:         # Checks if the word is a subject of the sentence or not.
                    map_sent_string =True


        elif item[2] == 'I-NP':
            result = process_word(item[0])
            string = string + " " + result

            if result in subjects:
                map_sent_string = True


    if string != "":
        concept = ' '.join(string.split())
        concepts.append(concept)
        if string not in concept_count:
            concept_count[string] = 1
        else:
            concept_count[string] += 1

        if num_times > 0 and map_sent_string == True:
            if string not in concept_sentences:
                concept_sentences[string] = [sent]
                num_times = num_times - 1
                map_sent_string = False
            else:
                if sent not in concept_sentences[string]:
                    concept_sentences[string].append(sent)
                    num_times = num_times - 1
                    map_sent_string = False



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
def prune_concepts_WordLevel(concept_count,pruned_concepts):
    freq_threshold = 50                                         # Words with frequency greater than the threshold are ignored
    df = pd.read_csv("/home/abhinav/BBCData.csv", header=None)
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

def Wikipedia_aritcle_matching(pruned_concepts,concept_wiki_article,final_wiki_concepts):
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



                # elif len(set(tokenized_article)-set(tokenized_key))==0:                  # if some wiki article is present as a substring in extracted concept, then also add it as a valid concept but do this only after verifying that exact match article dose not exist
                #     concept_wiki_article[key].append(article)
                #     if article not in final_wiki_concepts:
                #         final_wiki_concepts.append(article)


            if len(concept_wiki_article[key])==0:
                concept_wiki_article[key].append('')

        else:
            concept_wiki_article[key].append('')
        # print("Score:%d"%(max))

## For each pruned concept, there are Wikipedia Articles and sentences mapped. This function maps sentences to Wikipedia Articles
def wiki_article_sent_mapping(concept_wiki_article,concept_sentences,wiki_article_sentences):
    for concept in concept_wiki_article:
        if concept_wiki_article[concept][0] != '':
            if concept in concept_sentences:
                for key_concept in concept_wiki_article[concept]:
                    if key_concept not in wiki_article_sentences:
                        wiki_article_sentences[key_concept] = concept_sentences[concept]
                    else:
                        wiki_article_sentences[key_concept] = wiki_article_sentences[key_concept] + concept_sentences[concept]
            else:
                for key_concept in concept_wiki_article[concept]:
                    if key_concept not in wiki_article_sentences:
                        wiki_article_sentences[key_concept] = ['']

def main():

    with open('/home/abhinav/PycharmProjects/video_enrichment/text.txt', 'r') as myfile:
        text = myfile.read().replace('\n', '')

    concept_count = {}  # this stores number of time a concept has appeared in the text
    concept_sentences = {}  # this stores all the sentences for every concept in which they appeared
    pruned_concepts = {}  # this is pruned version of concept_count dictionary
    concept_wiki_article = {}  # this stores wikipedia articles for every concept
    wiki_article_sentences = {}  # this stores sentences corresponding to every wiki article
    final_wiki_concepts = []

    sentences = nltk.sent_tokenize(text)

    for sent in sentences:
        print(sent)
        orig_sent = sent
        sentence = sent.lower()                     # Lower Case the whole sentence
        sentence = p.sub('',sentence)                   # Removing anything enclosed within brackets



        ## TAGGING
        st_tag = StanfordPOSTagger(model_filename=eng_model_filename_pos, path_to_jar=my_path_to_pos_jar)
        tagged_sentence = st_tag.tag(word_tokenize(sentence))
        # print(tagged_sentence)

        ## ENTITY RECOGNITION
        # st_ner = StanfordNERTagger(model_filename=eng_model_filename_ner, path_to_jar=my_path_to_ner_jar)
        # print(st_ner.tag('Rami Eid is studying at Stony Brook University in NY'.split()))

        ## PARSING - to get subject from given sentence
        subjects = get_subject(sentence)
        # print(subjects)

        ## Chunking Using Regex
        regex_exps = ["NP: {<JJ|NN.?>*<NN.?>}", "NP: {<JJ|NN.?>*<NN.?><IN>?<JJ|NN.?>*<NN.?>}", "NP: {<JJ>*<NN.?>+}"] # Include the following pattern to count conjuctions "NP: {<JJ|NN.?>*<NN.?><CC>?<JJ|NN.?>*<NN.?>}"
        for grammar in regex_exps:
            IOB_tagged = chunking(tagged_sentence,grammar)
            remove_IOBtags(IOB_tagged,subjects,orig_sent,concept_count,concept_sentences)

    # print(concept_count)

    ## Prune concepts on word level using word frequency count from BBC corpus
    prune_concepts_WordLevel(concept_count,pruned_concepts)
    print("Pruned concepts are:",pruned_concepts)
    print("\n",concept_sentences)

    ## Identify Wikipedia articles(titles) that match concepts extracted from the text if Jaccard Similarity is one or if wikipedia title is a part of concept extracted
    Wikipedia_aritcle_matching(pruned_concepts,concept_wiki_article,final_wiki_concepts)
    print("\n",concept_wiki_article)
    print("\nFinal List Of Concepts:",final_wiki_concepts)

    wiki_article_sent_mapping(concept_wiki_article,concept_sentences,wiki_article_sentences)
    print("\n",(wiki_article_sentences))

if __name__ == "__main__":
    main()
