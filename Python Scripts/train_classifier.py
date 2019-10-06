# -*- coding: utf-8 -*-

from watson_developer_cloud import NaturalLanguageClassifierV1
import os
import json

#{
#  "url": "https://www.ibm.com/watson/developercloud/natural-language-classifier/api/v1/",
#  "username": "6f096082-40ee-4cac-9f28-2fcde55337db",
#  "password": "ODVHY5YNOQ05"
#}

# {
#   "classifier_id" : "6a2a04x217-nlc-35573",
#   "name" : "Definition_identification_1",
#   "language" : "en",
#   "created" : "2017-09-22T08:34:35.250Z",
#   "url" : "https://gateway.watsonplatform.net/natural-language-classifier/api/v1/classifiers/6a2a04x217-nlc-35573",
#   "status" : "Training",
#   "status_description" : "The classifier instance is in its training phase, not yet ready to accept classify requests"
# }
#==============================================================================
# Train the classifier with training data in train.csv
#==============================================================================

csvfilepath1 = '/home/abhinav/PycharmProjects/video_enrichment/dataset/train.csv'

nlc = NaturalLanguageClassifierV1(username="6f096082-40ee-4cac-9f28-2fcde55337db", password="ODVHY5YNOQ05")
with open(csvfilepath1, 'rb') as train:
   classifier = nlc.create(training_data=train, name='Definition_identification_1', language='en')
print (json.dumps(classifier, indent=2))


#==============================================================================
# Classify a test sample with the classifier using the classifier id
#==============================================================================
#nlc= NaturalLanguageClassifierV1(username="6f096082-40ee-4cac-9f28-2fcde55337db", password="ODVHY5YNOQ05")
#classes = nlc.classify('715dfax190-nlc-233', 'Can I transfer my existing bank account from one place to another? Do I need to undergo full KYC again?')
#print(json.dumps(classes, indent=2))
