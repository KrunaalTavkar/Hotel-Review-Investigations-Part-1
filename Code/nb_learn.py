#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:34:07 2020

@author: krunaaltavkar
"""

import os
import sys
import re
import operator
from collections import Counter
import math

class NBClassifier():
    
    def __init__(self):
        # Defining Stop Words and other constants for pattern detection for NBClassifier Object
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
                           "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                           "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
                           "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                           "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
                           "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                           "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                           "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
                           "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
                           "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
                           "most", "other", "some", "such", "nor", "not", "only", "own", "same", "so", "than", 
                           "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "", 'bed', 
                           'chicago', 'desk', 'get', 'great', 'hotel', 'like', 'location', 'nice', 'night', 'one', 
                           'room', 'rooms', 'service', 'staff', 'stay', 'stayed', 'time', 'us', 'would', 'even', 
                           'got', 'experience', 'went', 'breakfast', 'right']
        
        self.pos_pattern = 'positive'
        self.neg_pattern = 'negative'
        self.tru_pattern = 'truthful'
        self.dec_pattern = 'deceptive'
        self.train_set_pattern = r"(LICENSE|README|DS_Store)"
        self.conditional_probabilities = {}
        self.class_priors = []
    
    
    def get_training_documents(self, input_path):
        # Paths for data are read from command line
        train_file = input_path
        train_set = []
        for root, dirnames, files in os.walk(train_file):
            for file in files:
                file_str = root + '/' + file
                if not re.search(self.train_set_pattern, file_str):
                    train_set.append(file_str)
        
        return train_set
    
    
    def get_training_tokens_and_labels(self, train_set):
        cleaned_train_data = []
        train_sentiments = []
        train_tru_decep = []
        vocabulary = []
        
        for i in range(len(train_set)):
            f = open(train_set[i], 'r')
            data = f.read()
            data = data.lower()
            data = re.sub('[^a-z\s]+', " ", data)
            data = re.sub('(\s+)', " ", data)
            cleaned_train_data.append(data)
            if re.search(self.pos_pattern, train_set[i]):
                polarity = 1
            else:
                polarity = 0
            if re.search(self.tru_pattern, train_set[i]):
                truthful = 1
            else:
                truthful = 0
            train_sentiments.append(polarity)
            train_tru_decep.append(truthful)
        

        for i in range(len(cleaned_train_data)):
            all_words = [word for word in re.split("\s+", cleaned_train_data[i]) if word not in self.stop_words]
            for word in all_words:
                vocabulary.append(word)
            final_words = Counter(all_words)
            cleaned_train_data[i] = dict(final_words)
        
        vocabulary_counter = Counter(vocabulary)
        
        return cleaned_train_data, train_sentiments, train_tru_decep, vocabulary, vocabulary_counter
    
    
    def get_unique_tokens(self, cleaned_train_data):
        list_of_unique_words_in_training_corpus = []
        for dictionary in cleaned_train_data:
            for key in dictionary:
                list_of_unique_words_in_training_corpus.append(key)
        
        
        list_of_unique_words_in_training_corpus = list(set(list_of_unique_words_in_training_corpus))
        
        return list_of_unique_words_in_training_corpus
    
    
    def fit(self, cleaned_train_data, train_sentiments, train_tru_decep):
        list_of_unique_words_in_training_corpus = self.get_unique_tokens(cleaned_train_data)
        number_positive_truthful = len([x for x in range(len(train_sentiments)) if train_sentiments[x] == 1 and train_tru_decep[x] == 1])
        number_positive_deceptive = len([x for x in range(len(train_sentiments)) if train_sentiments[x] == 1 and train_tru_decep[x] == 0])
        number_negative_truthful = len([x for x in range(len(train_sentiments)) if train_sentiments[x] == 0 and train_tru_decep[x] == 1])
        number_negative_deceptive = len([x for x in range(len(train_sentiments)) if train_sentiments[x] == 0 and train_tru_decep[x] == 0])
        positive_truthful_tokens = 0
        negative_truthful_tokens = 0
        positive_deceptive_tokens = 0
        negative_deceptive_tokens = 0
        tokens_classes_count = {}
        train_docs = len(cleaned_train_data)
        for i in range(train_docs):
            for token in cleaned_train_data[i]:
                temp = {'positive-truthful' : 0,
                    'negative-truthful' : 0,
                    'positive-deceptive' : 0,
                    'negative-deceptive' : 0}
                if token not in tokens_classes_count:
                    if train_sentiments[i] == 1:
                        if train_tru_decep[i] == 1:
                            temp['positive-truthful'] += cleaned_train_data[i][token]
                            positive_truthful_tokens += cleaned_train_data[i][token]
                        else:
                            temp['positive-deceptive'] += cleaned_train_data[i][token]
                            positive_deceptive_tokens += cleaned_train_data[i][token]
                    else:
                        if train_tru_decep[i] == 1:
                            temp['negative-truthful'] += cleaned_train_data[i][token]
                            negative_truthful_tokens += cleaned_train_data[i][token]
                        else:
                            temp['negative-deceptive'] += cleaned_train_data[i][token]
                            negative_deceptive_tokens += cleaned_train_data[i][token]
                    tokens_classes_count[token] = temp
                
                else:
                    if train_sentiments[i] == 1:
                        if train_tru_decep[i] == 1:
                            positive_truthful_tokens += cleaned_train_data[i][token]
                            tokens_classes_count[token]['positive-truthful'] += cleaned_train_data[i][token]
                        else:
                            positive_deceptive_tokens += cleaned_train_data[i][token]
                            tokens_classes_count[token]['positive-deceptive'] += cleaned_train_data[i][token]
                    else:
                        if train_tru_decep[i] == 1:
                            negative_truthful_tokens += cleaned_train_data[i][token]
                            tokens_classes_count[token]['negative-truthful'] += cleaned_train_data[i][token]
                        else:
                            negative_deceptive_tokens += cleaned_train_data[i][token]
                            tokens_classes_count[token]['negative-deceptive'] += cleaned_train_data[i][token]
        
        class_priors = [number_positive_truthful/train_docs, number_negative_truthful/train_docs, number_positive_deceptive/train_docs, number_negative_deceptive/train_docs]
        conditional_probabilities = {}
        for word in list_of_unique_words_in_training_corpus:
            temp = {}
            pos_tru_value = (tokens_classes_count[word]['positive-truthful'] + 1) / (positive_truthful_tokens + len(list_of_unique_words_in_training_corpus))
            neg_tru_value = (tokens_classes_count[word]['negative-truthful'] + 1) / (negative_truthful_tokens + len(list_of_unique_words_in_training_corpus))
            pos_dec_value = (tokens_classes_count[word]['positive-deceptive'] + 1) / (positive_deceptive_tokens + len(list_of_unique_words_in_training_corpus))
            neg_dec_value = (tokens_classes_count[word]['negative-deceptive'] + 1) / (negative_deceptive_tokens + len(list_of_unique_words_in_training_corpus))
            
            temp['positive-truthful'] = pos_tru_value
            temp['negative-truthful'] = neg_tru_value
            temp['positive-deceptive'] = pos_dec_value
            temp['negative-deceptive'] = neg_dec_value
            conditional_probabilities[word] = temp
        
        self.class_priors = class_priors
        self.conditional_probabilities = conditional_probabilities
    
    
    def generate_model(self, class_priors, conditional_probabilities):
        priors = {'positive-truthful' : math.log(class_priors[0]),
            'negative-truthful' : math.log(class_priors[1]),
            'positive-deceptive' : math.log(class_priors[2]),
            'negative-deceptive' : math.log(class_priors[3])}

        write_rows = []
        prior_row = ['naivebayespriors', priors['positive-truthful'], priors['negative-truthful'], priors['positive-deceptive'], priors['negative-deceptive']]
        write_rows.append(prior_row)
        for word in conditional_probabilities:
            
            row = [word, conditional_probabilities[word]['positive-truthful'], 
                   conditional_probabilities[word]['negative-truthful'], 
                   conditional_probabilities[word]['positive-deceptive'], 
                   conditional_probabilities[word]['negative-deceptive']]
            
            write_rows.append(row)
        
        with open('nbmodel.txt', 'w') as f:
            
            for row in write_rows:
                my_str = ""
                for word in row:
                    my_str += str(word) + "\t"
                my_str += "\n"
                f.writelines(my_str)

if __name__== "__main__":
    input_path = sys.argv[1]
    nb_object = NBClassifier()
    train_set = nb_object.get_training_documents(input_path)
    cleaned_train_data, train_sentiments, train_tru_decep, vocabulary, vocabulary_counter = nb_object.get_training_tokens_and_labels(train_set)
    nb_object.fit(cleaned_train_data, train_sentiments, train_tru_decep)
    nb_object.generate_model(nb_object.class_priors, nb_object.conditional_probabilities)
    