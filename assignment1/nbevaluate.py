###################################################
# 
# Author - Arnab Sanyal (arnabsan@usc.edu)
# USC. Spring 2020
###################################################

import os
import sys

class NBEvaluate:
    def __init__(self, nb_output_file_path: str):
        self.spam_prec = None
        self.spam_rec = None
        self.spam_f1 = None
        self.ham_prec = None
        self.ham_rec = None
        self.ham_f1 = None
        self.nb_output_file_path = nb_output_file_path

    def calculate(self) -> None:
        content = None
        f = open(self.nb_output_file_path, mode='r')
        content = f.read().rstrip().split('\n')
        f.close()
        statistics = dict()
        statistics['ham:ham'] = 0
        statistics['ham:spam'] = 0
        statistics['spam:ham'] = 0
        statistics['spam:spam'] = 0
        for entry in content:
            verdict, path_var = entry.split()
            if ('ham' in path_var) and ('spam' == verdict):
                statistics['spam:ham'] += 1
            elif ('spam' in path_var) and ('spam' == verdict):
                statistics['spam:spam'] += 1
            elif ('ham' in path_var) and ('ham' == verdict):
                statistics['ham:ham'] += 1
            elif ('spam' in path_var) and ('ham' == verdict):
                statistics['ham:spam'] += 1
            else:
                pass
        if (statistics['ham:ham'] + statistics['ham:spam']) != 0:
            self.ham_prec = statistics['ham:ham'] / (statistics['ham:ham'] + statistics['ham:spam'])
        if (statistics['ham:ham'] + statistics['spam:ham']) != 0:
            self.ham_rec = statistics['ham:ham'] / (statistics['ham:ham'] + statistics['spam:ham'])
        if (self.ham_prec is not None) and (self.ham_rec is not None):
            self.ham_f1 = (2 * self.ham_rec * self.ham_prec) /(self.ham_rec + self.ham_prec)

        if (statistics['spam:spam'] + statistics['spam:ham']) != 0:
            self.spam_prec = statistics['spam:spam'] / (statistics['spam:spam'] + statistics['spam:ham'])
        if (statistics['spam:spam'] + statistics['ham:spam']) != 0:
            self.spam_rec = statistics['spam:spam'] / (statistics['spam:spam'] + statistics['ham:spam'])
        if (self.spam_prec is not None) and (self.spam_rec is not None):
            self.spam_f1 = (2 * self.spam_prec * self.spam_rec) / (self.spam_prec + self.spam_rec)

obj = NBEvaluate(sys.argv[1])
obj.calculate()
print('spam precision:', obj.spam_prec, '\nspam recall:', obj.spam_rec, '\nspam F1 score:', obj.spam_f1, '\nham precision:', obj.ham_prec, '\nham recall:', obj.ham_rec, '\nham F1 score:', obj.ham_f1)