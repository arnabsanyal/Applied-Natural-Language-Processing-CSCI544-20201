###################################################
# 
# Author - Arnab Sanyal (arnabsan@usc.edu)
# USC. Spring 2020
###################################################

from hw2_corpus_tool import *
import pycrfsuite
import os
import sys
import random


class BaselineTagger:
    def __init__(self, train_path: str, test_path: str, out_file: str):
        self.train_set_fnames = []
        self.test_set_fnames = []
        self.train_path = train_path
        self.test_path = test_path
        self.out_file = out_file
        self.model = dict()

    def segregate_data_loader(self) -> None:
        total_set = [name for name in os.listdir(self.test_path) if os.path.isfile(os.path.join(self.test_path, name))]
        f_num = len(total_set)
        random.shuffle(x=total_set, random=None)
        delim = int(f_num >> 2)
        self.train_set_fnames = total_set[delim:]
        self.test_set_fnames = total_set[:delim]
        self.test_set_fnames.sort()
        del total_set
        return None
    
    def data_loader_init(self) -> None:
        self.train_set_fnames = [name for name in os.listdir(self.train_path) if os.path.isfile(os.path.join(self.train_path, name))]
        self.test_set_fnames = [name for name in os.listdir(self.test_path) if os.path.isfile(os.path.join(self.test_path, name))]
        self.test_set_fnames.sort()
        return None

    def model_feature_synthesizer(self) -> None:
        engineered_features, labels = [], []
        currentSpeaker, previousSpeaker = None, None
        for f_name in self.train_set_fnames:
            full_name = os.path.join(self.train_path, f_name)
            csvReader = get_utterances_from_filename(full_name)
            first_utt = True

            for thisUtterance in csvReader:
                feature = []
                currentSpeaker = thisUtterance.speaker
                feature = (feature + ["n"]) if (currentSpeaker != previousSpeaker) else (feature + ["y"])
                previousSpeaker = currentSpeaker
                if first_utt:
                    feature += ["y"]
                    first_utt = False
                else:
                    feature += ["n"]
                act_tag = thisUtterance.act_tag if(thisUtterance.act_tag is not None) else None
                if thisUtterance.pos is not None:
                    for postag in thisUtterance.pos:
                        if postag.token is not None:
                            feature += ["TOKEN_" + postag.token.lower()]
                        else:
                            feature += ["ERR"]
                        if postag.pos is not None:
                            feature += ["POS_" + postag.pos.upper()]
                        else:
                            feature += ["ERR"]

                engineered_features += [feature]
                labels += [act_tag]
        self.model["trainer"] = pycrfsuite.Trainer(verbose=False)

        self.model["trainer"].set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        self.model["trainer"].append(engineered_features, labels)
        self.model["trainer"].train('swbd-damsl-1997-baseline_csv.crfsuite')
        return None

    def model_tester(self) -> None:
        self.model["num"] = 0
        self.model["denom"] = 0
        self.model["tagger"] = pycrfsuite.Tagger()
        self.model["tagger"].open('swbd-damsl-1997-baseline_csv.crfsuite')
        file_writer = ""
        currentSpeaker, previousSpeaker = None, None
        for f_name in self.test_set_fnames:
            full_name = os.path.join(self.test_path, f_name)
            csvReader = get_utterances_from_filename(full_name)
            engineered_features, true_labels = [], []
            first_utt = True

            for thisUtterance in csvReader:
                feature = []
                currentSpeaker = thisUtterance.speaker
                feature = (feature + ["n"]) if (currentSpeaker != previousSpeaker) else (feature + ["y"])
                previousSpeaker = currentSpeaker
                if first_utt:
                    feature += ["y"]
                    first_utt = False
                else:
                    feature += ["n"]
                act_tag = thisUtterance.act_tag if(thisUtterance.act_tag is not None) else None
                if thisUtterance.pos is not None:
                    for postag in thisUtterance.pos:
                        if postag.token is not None:
                            feature += ["TOKEN_" + postag.token.lower()]
                        else:
                            feature += ["ERR"]
                        if postag.pos is not None:
                            feature += ["POS_" + postag.pos.upper()]
                        else:
                            feature += ["ERR"]

                engineered_features += [feature]
                true_labels += [act_tag]
            predicted_labels = self.model["tagger"].tag(engineered_features)
            file_writer += "\n".join(predicted_labels) + "\n\n"
            if len(sys.argv) == 5:
                self.model["denom"] += len(true_labels)
                for i in range(len(true_labels)):
                    if true_labels[i] is None:
                        self.model["denom"] -= 1
                        continue
                    if true_labels[i] == predicted_labels[i]:
                        self.model["num"] += 1
                
        out_file = open(self.out_file, "w")
        out_file.write(file_writer)
        out_file.close()
        return None



train_path, test_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
obj = BaselineTagger(train_path, test_path, out_file)
if train_path == test_path:
    obj.segregate_data_loader()
else:
    obj.data_loader_init()

obj.model_feature_synthesizer()
obj.model_tester()
if len(sys.argv) == 5:
    print("Accuracy of BaseLine Tagger Object on SWBD-DAMSL is %f %%" %(float(100 * obj.model["num"])/obj.model["denom"]))