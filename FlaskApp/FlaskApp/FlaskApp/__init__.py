from __future__ import division
import nltk
import re
import string as string_lib
import pandas as pd
import numpy
import networkx as nx
import math
import wikipedia
import pickle
import key_concept_extraction
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
from flask import Flask, render_template, url_for, request


## Global Variables
p = re.compile('\([^()]*\)')                             # won't work in cases where bracket in bracket
reg_punc = re.compile('[%s]' % re.escape(string_lib.punctuation))
stop = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

## Global Scope Functions
tokenize = lambda doc: doc.lower().split(" ")

# Loaction of all jar files
stanford_pos_dir = '/home/abhinavj/stanford-postagger-full-2017-06-09/'
eng_model_filename_pos = stanford_pos_dir + 'models/english-left3words-distsim.tagger'
my_path_to_pos_jar= stanford_pos_dir + 'stanford-postagger.jar'

stanford_parser_dir = '/home/abhinavj/stanford-parser-full-2017-06-09/'
eng_model_filename_parser = stanford_parser_dir + "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz"
my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.8.0-models.jar"
my_path_to_parser_jar = stanford_parser_dir + "stanford-parser.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=my_path_to_parser_jar,path_to_models_jar=my_path_to_models_jar)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template("rishi.html")
#    return render_template("index.html", title = "Welcome", paragraph = "I am Rishi and I am so cool")

@app.route('/give_me_text/', methods=['GET', 'POST'])
def give_me_text():
    
    if request.form['action'] == 'Extract Concepts':
        exec(compile(open('clearcontents.py', "rb").read(), 'clearcontents.py', 'exec'))
        xyz = request.form["input_text"]
        print (xyz)
        with open('fetched_url_download','wb') as fp1:
            pickle.dump(xyz, fp1)

        exec(compile(open('execfile.py', "rb").read(), 'execfile.py', 'exec'))
        print ("File run success")
        return render_template("results.html")
    # if request.form['action'] == 'Extract Prerequistes':
    #     tts = request.form['exampleFormControlSelect1']
    #     flash(str(tts)+'is being selected')
    #     return render_template("rishi.html")

@app.route("/test/" , methods=['GET', 'POST'])
def test():
    print ("Entered test")
    if request.form['action'] == 'Extract Prerequisites':
        print ("Entered if of test")
        tts = request.form['exampleFormControlSelect1']
        print (str(tts)+' is being selected')
        with open('selected_concept','wb') as fp1:
            pickle.dump(tts, fp1)
        return render_template("results.html")

    elif request.form['action'] == 'Extract Definition':
        print ("Entered if of test")
        exec(compile(open('clearcontents_def.py', "rb").read(), 'clearcontents_def.py', 'exec'))
        # execfile('clearcontents_def.py')
        tts = request.form['exampleFormControlSelect1']
        print (str(tts)+' is being selected')
        with open('selected_concept','wb') as fp1:
            pickle.dump(tts, fp1)
        exec(compile(open('execfile_def.py', "rb").read(), 'execfile_def.py', 'exec'))
        # execfile("execfile_def.py")
        print ("File run success")
        return render_template("results1.html")

    elif request.form['action'] == 'Extract Application':
        print ("Entered if of test")
        exec(compile(open('clearcontents_app.py', "rb").read(), 'clearcontents_app.py', 'exec'))
        # execfile('clearcontents_app.py')
        tts = request.form['exampleFormControlSelect1']
        print (str(tts)+' is being selected')
        with open('selected_concept','wb') as fp1:
            pickle.dump(tts, fp1)
        exec(compile(open('execfile_def.py', "rb").read(), 'execfile_def.py', 'exec'))
        # execfile("execfile_app.py")
        print ("File run success")
        return render_template("results2.html")

    elif request.form['action'] == 'Submit Feedback':
        print ("Entered if of test")
        pq = request.form['pq']
        print (str(pq)+' is being selected')
        der = request.form['der']
        print (str(der)+' is being selected')
        aer = request.form['aer']
        print (str(aer)+' is being selected')

        # Only to intialize the counters
        # pq_ctr = {'ba':0,'a':0,'g':0,'e':0}
        # with open('prereq_graph','wb') as fp:
        #     pickle.dump(pq_ctr, fp)

        # der_ctr = {'op1':0,'op2':0,'op3':0,'op4':0}
        # with open ('def_enrichment', 'wb') as fp:
        #     pickle.dump(der_ctr, fp)

        # aer_ctr = {'op1':0,'op2':0,'op3':0,'op4':0}
        # with open ('app_enrichment', 'wb') as fp:
        #     pickle.dump(aer_ctr, fp)

        # Checking for Prereq Graph counter
        with open ('prereq_graph', 'rb') as fp:
            pq_ctr = pickle.load(fp)

        if pq == 'ba':
            pq_ctr['ba'] += 1
        elif pq == 'a':
            pq_ctr['a'] += 1
        elif pq == 'g':
            pq_ctr['g'] += 1
        elif pq == 'e':
            pq_ctr['e'] += 1

        with open('prereq_graph','wb') as fp1:
            pickle.dump(pq_ctr, fp1)

        # Checking for Definiton Enrichment counter
        with open ('def_enrichment', 'rb') as fp:
            der_ctr = pickle.load(fp)

        if der == 'op1':
            der_ctr['op1'] += 1
        elif der == 'op2':
            der_ctr['op2'] += 1
        elif der == 'op3':
            der_ctr['op3'] += 1
        elif der == 'op4':
            der_ctr['op4'] += 1

        with open('def_enrichment','wb') as fp1:
            pickle.dump(der_ctr, fp1)

        # Checking for Application Enrichment counter
        with open ('app_enrichment', 'rb') as fp:
            aer_ctr = pickle.load(fp)

        if aer == 'op1':
            aer_ctr['op1'] += 1
        elif aer == 'op2':
            aer_ctr['op2'] += 1
        elif aer == 'op3':
            aer_ctr['op3'] += 1
        elif aer == 'op4':
            aer_ctr['op4'] += 1

        with open('app_enrichment','wb') as fp1:
            pickle.dump(aer_ctr, fp1)

        return render_template("rishi.html")

        # Submit Feedback


#@app.route('/page2', methods=['GET', 'POST'])

# def homepage():
#     return render_template("results.html")
# #    return render_template("index.html", title = "Welcome", paragraph = "I am Rishi and I am so cool")

#@app.route('/give_me_text2', methods=['GET', 'POST'])


 #    elif request.form['action'] == 'Extract Definition':
 #        abc = request.form["name of that thing"]
 #        print abc
 #    return render_template("rishi.html")

 #    elif request.form['action'] == 'Extract Application':
 #        abc = request.form["name of that thing"]
 #        print abc
 #    return render_template("rishi.html")

if __name__ == "__main__":
    app.run(debug=True)
