###################################################
# 
# Author - Arnab Sanyal (arnabsan@usc.edu)
# USC. Spring 2020
###################################################

import json
import os
import sys
import math

class NBClassify:
    def __init__(self, model_path: str, file_path: str):
        self.model = None
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        self.file_path = file_path
        self.classification_result = ''
        self.model['vocabulary'] = set(self.model['vocabulary'])

    def driver(self) -> None:
        for root, directories, filenames in os.walk(os.path.join(os.getcwd(), self.file_path)):
            for filename in filenames:
                path_var = os.path.join(root, filename)
                if path_var[-4:] == '.txt':
                    self.decide(path_var)

    def decide(self, path_var: str) -> None:
        content = None
        f = open(path_var, mode='r', encoding='latin1')
        content = f.read().split(' ')
        f.close()
        content = [token.rstrip() for token in content]
        label = ''
        spam_score = self.model['p_spam']
        ham_score = self.model['p_ham']
        for token in content:
            if token in self.model['vocabulary']:
                if token in self.model['p_hamTokens']:
                    ham_score += self.model['p_hamTokens'][token]
                else:
                    ham_score += math.log(1.0/(self.model['count_ham'] + self.model['vocabulary_size']))
                if token in self.model['p_spamTokens']:
                    spam_score += self.model['p_spamTokens'][token]
                else:
                    spam_score += math.log(1.0/(self.model['count_spam'] + self.model['vocabulary_size']))

        label = 'ham' if (ham_score > spam_score) else 'spam'
        self.classification_result += (label + '\t' + path_var + '\n')

    def generateResultFile(self) -> None:
        f = open('nboutput.txt', 'w')
        f.write(self.classification_result)
        f.close()

obj = NBClassify('./nbmodel.txt', sys.argv[1])
obj.driver()
obj.generateResultFile()