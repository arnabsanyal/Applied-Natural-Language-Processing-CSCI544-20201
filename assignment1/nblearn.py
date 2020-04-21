###################################################
# 
# Author - Arnab Sanyal (arnabsan@usc.edu)
# USC. Spring 2020
###################################################

import sys
import os
import json
import math

class NBLearn:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.count_ham = 0
        self.count_spam = 0
        self.count_hamTokens = dict()
        self.count_spamTokens = dict()
        self.vocabulary = None
        self.vocabulary_size = 0

    def dataLoader(self) -> None:
        subdirs = os.listdir(self.file_path)

        for subdir in subdirs:
            if subdir[0] == '.':
                continue
            vPath_prefix = os.path.join(self.file_path, subdir, 'spam')
            emails = os.listdir(vPath_prefix)
            for email in emails:
                f = open(os.path.join(vPath_prefix, email), mode='r', encoding='latin1')
                content = f.read().split(' ')
                f.close()
                content = [token.rstrip() for token in content]
                self.count_spam += len(content)
                for token in content:
                    if token in self.count_spamTokens:
                        self.count_spamTokens[token] += 1
                    else:
                        self.count_spamTokens[token] = 1
                

        for subdir in subdirs:
            if subdir[0] == '.':
                continue
            vPath_prefix = os.path.join(self.file_path, subdir, 'ham')
            emails = os.listdir(vPath_prefix)
            for email in emails:
                f = open(os.path.join(vPath_prefix, email), mode='r', encoding='latin1')
                content = f.read().split(' ')
                f.close()
                content = [token.rstrip() for token in content]
                self.count_ham += len(content)
                for token in content:
                    if token in self.count_hamTokens:
                        self.count_hamTokens[token] += 1
                    else:
                        self.count_hamTokens[token] = 1

        self.vocabulary = list(set(list(self.count_hamTokens.keys()) + list(self.count_spamTokens.keys())))
        self.vocabulary_size = len(self.vocabulary)

    def modelCreator(self) -> None:
        model = dict()
        model['vocabulary'] = self.vocabulary
        model['vocabulary_size'] = self.vocabulary_size
        model['count_ham'] = self.count_ham
        model['count_spam'] = self.count_spam
        model['p_ham'] = math.log(self.count_ham/(self.count_ham + self.count_spam))
        model['p_spam'] = math.log(1.0 - model['p_ham'])
        model['p_hamTokens'] = dict()
        model['p_spamTokens'] = dict()

        for token in self.count_hamTokens:
            model['p_hamTokens'][token] = math.log((self.count_hamTokens[token] + 1)/(self.count_ham + self.vocabulary_size))
        for token in self.count_spamTokens:
            model['p_spamTokens'][token] = math.log((self.count_spamTokens[token] + 1)/(self.count_spam + self.vocabulary_size))

        with open('nbmodel.txt', 'w') as f:
            json.dump(model, f)


obj = NBLearn(sys.argv[1])
obj.dataLoader()
obj.modelCreator()