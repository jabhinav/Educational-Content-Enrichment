"""
Embedding Dim : 300
LSTM output Dim: 300
Droput Keep prob :0.80 (Drop out units :0.20)
Top N words : 1000
Sequence Length : 500
Batch Size :32
Number of Epochs : 2
"""

import numpy as np
import pandas as pd
import os
import pickle
import csv
import re
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from keras.layers.convolutional import Conv1D,MaxPooling1D
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score,recall_score
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate

## Path to stanford POS tagger
# stanford_pos_dir = '/home/abhinav/NLP/softwares/stanford-postagger-full-2017-06-09/'
## For local server
stanford_pos_dir = '/home/abhinavj/stanford-postagger-full-2017-06-09/'
eng_model_filename_pos = stanford_pos_dir + 'models/english-left3words-distsim.tagger'
my_path_to_pos_jar= stanford_pos_dir + 'stanford-postagger.jar'
st_tag = StanfordPOSTagger(model_filename=eng_model_filename_pos, path_to_jar=my_path_to_pos_jar)

## GLobal Variables
most_common_tokens = []
index_word = {}
top_N_words = 1000
model_type ='vanilla_CNN_2Layer.h5'
## For local machine
# folderpath = '/home/abhinav/PycharmProjects/video_enrichment/dataset'
## For local server
folderpath = '/home/abhinavj/video_enrichment/dataset'

# fix random seed for reproducibility, initialize our random number generator to ensure our results are reproducible
np.random.seed(7)

def write_csv(data, fname):
    with open(os.path.join(folderpath, fname), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

def save_dic(loc,dic):
    pickle_out = open(loc,"wb")
    pickle.dump(dic,pickle_out)
    pickle_out.close()

def load_dic(loc):
    pickle_in = open(loc, "rb")
    dic = pickle.load(pickle_in)
    return dic

class text_CNN:
    def __init__(self,embeddings_index=None,EMBEDDING_DIM = 300,MAX_SEQ_LENGTH = 500, MAX_NB_WORDS=1000,nb_lstm=300):
        self.embeddings_index = embeddings_index
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQ_LENGTH
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.nb_lstm = nb_lstm

    def word_index_gen(self,pos_lines, neg_lines):

        # vectorize the text samples into a 2D integer tensor; word indexing is done in decreasing order of word frequency
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)  # need it for converting text into index sequence, set lower = False if you don't want to set tokens to lower case

        complete_set = pos_lines + neg_lines
        self.tokenizer.fit_on_texts(complete_set)
        pos_sequences = self.tokenizer.texts_to_sequences(pos_lines)
        neg_sequences = self.tokenizer.texts_to_sequences(neg_lines)

        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))
        print('Each sentence is encoded into a sequence of word indices of top %i frequent words ' % self.MAX_NB_WORDS)

        pos_data = pad_sequences(pos_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        neg_data = pad_sequences(neg_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        self.data = np.concatenate((pos_data, neg_data), axis=0)

        pos_labels = np.ones(len(pos_lines))
        neg_labels = np.zeros(len(neg_lines))
        self.labels = np.concatenate((pos_labels, neg_labels), axis=0)
        labels_binary_matrix = to_categorical(self.labels)

        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', labels_binary_matrix.shape)


    ## Preparing the embedding layer
    def embedding_layer_prep(self):
        print("Vectorizing Text Dataset")
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        num_words = min(self.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((num_words + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.MAX_NB_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(num_words + 1, self.EMBEDDING_DIM, weights=[embedding_matrix],input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)


    def define_CNN_model(self):

        model = Sequential()
        model.add(self.embedding_layer)
        model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))                                                # Since, this a binary classification, we need dense layer to output single number which is either 0 or 1 (therefore sigmoid activation)

        ## Or else, Make the Dense layer output two values with activation = 'anything' and then use softmax layer like this-
        # model.add(Dense(2))
        # model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # metrics: list of metrics to be evaluated by the model during training and testing.
        return model

    def cross_validate(self,num_splits=10):
        self.num_splits = num_splits
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        data = self.data[indices]
        labels = self.labels[indices]

        self.embedding_layer_prep()
        kfold = StratifiedKFold(n_splits=self.num_splits, shuffle=True,random_state=7)
        model = KerasClassifier(build_fn=self.define_CNN_model, epochs=10, batch_size=32, verbose=1)
        scoring = {'acc': 'accuracy',
                   'prec': 'precision',
                   'rec': 'recall',
                   'F1':'f1'}
        scores = cross_validate(model, data, labels,scoring=scoring, cv=kfold)
        print(scores)
        print("Accuracies:%f"%np.mean(scores['test_acc']))

        ## Precision, Recall and F1 measure need label Binarizer "lb = sklearn.preprocessing.LabelBinarizer()"
        print("Precision:%f"%np.mean(scores['test_prec']))
        print("Recall:%f" % np.mean(scores['test_recall']))
        print("F1:%f" % np.mean(scores['test_F1']))

    def final_model(self):
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        data = self.data[indices]
        labels = self.labels[indices]
        self.embedding_layer_prep()
        self.model = self.define_LSTM_model()
        print(self.model.summary())
        history = self.model.fit(data, labels, epochs=10, batch_size=32, verbose=1)



    def save_tokeniser(self,dir):
        fname = os.path.join(dir,'tokenizer.pickle')
        with open(fname,'wb') as handle:
            pickle.dump(self.tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

    def save_weights(self,dir):
        fname = os.path.join(dir,model_type)
        self.model.save_weights(fname)

    def load_model(self,model_dir):
        self.embedding_layer_prep()
        self.define_CNN_model()
        fname = os.path.join(model_dir, model_type)
        self.model.load_weights(fname)

    def load_tokeniser(self,dir):
        pickle_in = open(os.path.join(dir,'tokenizer.pickle'), "rb")
        self.tokenizer = pickle.load(pickle_in)
        self.word_index = self.tokenizer.word_index

    def test_text_proc(self,test_lines):
        self.test_lines = test_lines
        pos_test_seq = self.tokenizer.texts_to_sequences(self.test_lines)
        self.pos_test_data = np.array(pad_sequences(pos_test_seq, maxlen=self.MAX_SEQUENCE_LENGTH))
        self.pos_test_labels = np.ones(len(test_lines))

    def predict_scores(self):
        scores = self.model.evaluate(self.pos_test_data,self.pos_test_labels,batch_size=32,verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def predic_classes(self):
        predictions = self.model.predict(self.pos_test_data,batch_size=32,verbose=1)
        print("PREDICTIONS:")
        for i in range(len(self.test_lines)):
            print("%i: %s"%(i+1,self.test_lines[i]))
            print("PREDICTED:",predictions[i],  "EXPECTED:",self.pos_test_labels[i])

def main():

    ## Run for once to save embeddings dictionary
    # parse_GLOVE()
    path1 = os.path.join(folderpath,"Definition")
    path2 = os.path.join(folderpath,'Application_Extraction')
    pos_train = os.path.join(path1,'pos_train.csv')
    neg_train = os.path.join(path1,'neg_train.csv')
    with open(pos_train,'r') as f:
        pos_lines = f.readlines()
        print("Number of Positive Samples:%i"%len(pos_lines))
    with open(neg_train,'r') as f:
        neg_lines = f.readlines()
        print("Number of Negative Samples:%i" % len(neg_lines))

    ## DO this step for both cases : training + testing or just testing
    embeddings = load_dic(os.path.join(folderpath, "embeddings_index_42B_300d.pickle"))
    CNN_obj = text_CNN(embeddings_index=embeddings)


    ## STEP-1 : Token transformation [Optional]
    # token_transformation(1000,pos_lines,neg_lines)

    model_dir = os.path.join(path1, 'Model')


    ## STEP-2 : Preparing Text data
    print("STEP-2 : PREPROCESSING TEXT DATA")
    CNN_obj.word_index_gen(pos_lines,neg_lines)
    # CNN_obj.save_tokeniser(model_dir)


    ## STEP-3 : Defining the model and training
    print("STEP-3 : TRAINING")
    CNN_obj.cross_validate(num_splits=10)
    CNN_obj.final_model()
    CNN_obj.save_weights(model_dir)

    ## STEP-4 : Testing
    # print("MODEL LOADING...")

    ## Run the following two functions only when testing is being done standalone
    # CNN_obj.load_tokeniser(model_dir)
    # CNN_obj.load_model(model_dir)

    with open(os.path.join(path1,'test_positive.csv'),'r') as f:
        test_lines = f.readlines()
        print("TESTING number of Positive Samples:%i"%len(test_lines))
    CNN_obj.test_text_proc(test_lines)
    CNN_obj.predict_scores()
    # CNN_obj.predic_classes()




if __name__ == '__main__':
    main()