#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:16:26 2020

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
                           "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",                            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
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
        
        self.nb_model = {}
    
    def get_model(self):
        with open('nbmodel.txt', 'r') as document:
            nb_model = {}
            for line in document:
                line = line.split()
                if not line:  # empty line?
                    continue
                line[1:] = [float(i) for i in line[1:]]
                nb_model[line[0]] = line[1:]
        
        self.nb_model = nb_model
    
    def get_test_documents(self, input_path):
        test_set = []
        test_set_pattern = r"(file)"
        for root, dirnames, files in os.walk(input_path):
            for file in files:
                file_str = root + '/' + file
                if re.search(test_set_pattern, file_str):
                    test_set.append(file_str)
        
        return test_set
    
    def get_test_tokens(self, test_set):
        cleaned_test_data = []
        for i in range(len(test_set)):
            f = open(test_set[i], 'r')
            data = f.read()
            data = data.lower()
            data = re.sub('[^a-z\s]+', " ", data)
            data = re.sub('(\s+)', " ", data)
            cleaned_test_data.append(data)
        
        for i in range(len(cleaned_test_data)):
            all_words = [word for word in re.split("\s+", cleaned_test_data[i]) if word not in self.stop_words]
            all_words = Counter(all_words)
            cleaned_test_data[i] = dict(all_words)
        
        return cleaned_test_data

    def predict(self, test_set, cleaned_test_data):
        classes = ['positive-truthful', 'negative-truthful', 'positive-deceptive', 'negative-deceptive']
        class_priors = self.nb_model['naivebayespriors']
        output_data = []
        counter_val = 0
        for doc in cleaned_test_data:
            temp = {'positive-truthful' : class_priors[0],
                    'negative-truthful' : class_priors[1],
                    'positive-deceptive' : class_priors[2],
                    'negative-deceptive' : class_priors[3]}
            for token in doc:
                for cl in range(len(classes)):
                    if token in self.nb_model:
                        temp[classes[cl]] += math.log(self.nb_model[token][cl])
                        
            counts = Counter(temp)
        #    print(counts)
            top_2 = counts.most_common(1)
        #    print(top_2)
            output_row = []
            labels = [i[0] for i in top_2]
            if 'positive-truthful' in labels:
                output_row.append('truthful')
                output_row.append('positive')
            elif 'negative-truthful' in labels:
                output_row.append('truthful')
                output_row.append('negative')
            elif 'negative-deceptive' in labels:
                output_row.append('deceptive')
                output_row.append('negative')
            else:
                output_row.append('deceptive')
                output_row.append('positive')
            output_row.append(test_set[counter_val])
            counter_val += 1
            output_data.append(output_row)
        
        return output_data

    def generate_output(self, output_data):
        with open('nboutput.txt', 'w') as f:
            for row in output_data:
                my_str = ""
                for word in row:
                    my_str += str(word) + "\t"
                my_str += "\n"
                f.writelines(my_str)

if __name__== "__main__":
    input_path = sys.argv[1]
    nb_object = NBClassifier()
    nb_object.get_model()
    test_set = nb_object.get_test_documents(input_path)
    cleaned_test_data = nb_object.get_test_tokens(test_set)
    output_data = nb_object.predict(test_set, cleaned_test_data)
    nb_object.generate_output(output_data)