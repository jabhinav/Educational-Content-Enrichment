import numpy as np
import pandas as pd
import os
import pickle
import csv
import re
import wikipedia
import nltk
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.layers.convolutional import Conv1D, MaxPooling1D
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate

top_N_words = 1000
model_type_APP = 'CNN_LSTM_dropout_weights.h5'
## For local machine
# folderpath = '/home/abhinav/PycharmProjects/video_enrichment/dataset'
## For local server
folderpath = '/home/abhinavj/video_enrichment/dataset'

# fix random seed for reproducibility, initialize our random number generator to ensure our results are reproducible
np.random.seed(7)

def load_dic(loc):
    pickle_in = open(loc, "rb")
    dic = pickle.load(pickle_in)
    return dic

class testing:
    def __init__(self, dir=None, embeddings_index=None, EMBEDDING_DIM=300, MAX_SEQ_LENGTH=500, MAX_NB_WORDS=1000,
                 nb_lstm=300):
        self.dir = dir
        pickle_in = open(os.path.join(self.dir, 'tokenizer.pickle'), "rb")
        self.tokenizer = pickle.load(pickle_in)
        self.word_index = self.tokenizer.word_index
        self.embeddings_index = load_dic(os.path.join(dir, "embeddings_index_42B_300d.pickle"))
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQ_LENGTH
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.nb_lstm = nb_lstm

    def embedding_layer_prep(self):
        print("Vectorizing Text Dataset")
        print('Preparing embedding matrix.')

        num_words = min(self.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((num_words + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.MAX_NB_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(num_words + 1, self.EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)

    def define_LSTM_model(self):

        model = Sequential()
        model.add(self.embedding_layer)
        model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(self.nb_lstm, activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_model(self):
        self.embedding_layer_prep()
        self.model = self.define_LSTM_model()

    def test_text_proc(self,test_line):
        test_seq = self.tokenizer.texts_to_sequences(test_line) # the tokenizer automatically make the characters lower case
        test_data = np.array(pad_sequences(test_seq, maxlen=self.MAX_SEQUENCE_LENGTH))
        return test_data

    def predic_classes(self,sentence):
        self.sentence = sentence
        test_data = self.test_text_proc(self.sentence)
        predictions = self.model.predict_classes(test_data,batch_size=None,verbose=1)
        return predictions


def remove_non_ascii(text):
    s = ''.join([i if ord(i) < 128 else '' for i in text])
    #    return s.replace('\n','').replace('\r','').replace('\t','');
    return ' '.join(s.split())


def main():

    with open('selected_concept', 'rb') as fp:
        selected_key_concept = pickle.load(fp)
    with open('final_concepts_mapping', 'rb') as fp:
        wiki_article_sentences = pickle.load(fp)

    sentences = wiki_article_sentences[selected_key_concept]
    dir = os.path.join(folderpath,"Application_Extraction/Model")
    p = re.compile('\([^()]*\)')
    test_obj = testing(dir)
    test_obj.load_model()

    check_app_in_text = False
    check_app_in_wiki = False

    applications = []
    for sent in sentences:
        orig_sent = sent
        if 10 < len(sent):  # To remove sentences with single word or very few words or which are empty
            processed_1 = p.sub('', sent)
            processed_2 = remove_non_ascii(processed_1)
            processed_3 = processed_2.replace('[', '').replace(']', '').replace(';', ',')
            result = test_obj.predic_classes([processed_3])

            if result[0][0]==1:
                applications.append(orig_sent)
                check_app_in_text = True
                print("Application found")

    if check_app_in_text == False:
        applications = []
        obj = wikipedia.WikipediaPage(selected_key_concept)
        applications_text = obj.section("Applications")
        if applications_text != None:
            wiki_sentences = nltk.sent_tokenize(applications_text)
            for sent in wikipedia_sentences:
                orig_sent = sent
                if 10 < len(sent):  # To remove sentences with single word or very few words or which are empty
                    processed_1 = p.sub('', sent)
                    processed_2 = remove_non_ascii(processed_1)
                    processed_3 = processed_2.replace('[', '').replace(']', '').replace(';', ',')
                    result = test_obj.predic_classes([processed_3])

                    if result[0][0]==1:
                        check_app_in_wiki = True
                        applications.append(orig_sent)

    if len(applications) == 0:
        applications = ["No Applications Found"]

    print(applications)

    with open('app_in_text', 'wb') as fp:
        pickle.dump(check_app_in_text, fp)

    with open('app_in_wiki', 'wb') as fp:
        pickle.dump(check_app_in_wiki, fp)

    with open('applications', 'wb') as fp:
        pickle.dump(applications, fp)


if __name__ == '__main__':
    main()
