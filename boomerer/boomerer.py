#!/us/bin/env python
#-*- coding:utf-8 -*-
##
# Boomer.py => class of calling all Boosted Rules method
##
# Date: 2025/06/05
# Author: Hao Hu

#======================================
import numpy as np
import sys, os
import resource
import pickle
import copy
import math

from mlrl.boosting import Boomer
import mlrl.common.cython.rule_model as rulemodel

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .encode import MXEncoderBR
from .explain import MXExplainerBR, SATExplainerBR

sys.path.append("..")
from data import Data
from options import Options

#=======================================
class Boomerer(object):
    """
        The main class to train Boosted Rules (Boomer) model
    """
    
    def __init__(self, options, from_data=None, from_model=None,
                from_encoding=None):
        """
            Constructor
        """

        assert from_data or from_model or from_encoding, 'At least one input file should be specified'

        self.init_stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.init_ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

        # saving command-line options
        self.options = options
        self.seed = self.options.seed
        np.random.seed(self.seed)

        # from_data is to dataset used to train/test
        if from_data:
            self.use_categorical = self.options.use_categorical
            # saving data
            self.data = from_data
            dataset = np.asarray(self.data.samps, dtype=np.float32)

            # split data into X and y
            self.feature_names = self.data.names[:-1]
            self.nb_features = len(self.feature_names)

            self.X = dataset[:, 0:self.nb_features]
            self.Y = dataset[:, self.nb_features]
            self.num_class = len(set(self.Y))
            self.target_name = list(range(self.num_class))
            self.num_instance = len(self.Y)

            # fixing the issue with labels starting from 1 rather than from 0
            le = LabelEncoder()
            self.Y = le.fit_transform(self.Y)

            # preparing the parameters for Boomer
            # update: 250701: considering max_rule as the number of rules for each class separately, not all.
            if self.num_class > 2:
                param_dict = {'max_rules': self.options.n_estimators * self.num_class}
            else:
                param_dict = {'max_rules': self.options.n_estimators}

            # update: 250924: add the number of maximum conditions in each rule => useful as max_depth
            param_dict['rule_induction'] = 'top-down-greedy{max_conditions=' + str(self.options.maxdepth) + '}'
            # update: 250926: add parameter 'round-robin' to make the number of rules learned 
            # for each class is one by one
            param_dict['output_sampling'] = 'round-robin'

            # param_dict['binary_predictor'] = 'output-wise'
            param_dict['binary_predictor'] = 'example-wise'
    
            # param_dict['loss'] = 'logistic-decomposable'
            # param_dict['loss'] = 'logistic-non-decomposable'
            param_dict['loss'] = 'squared-error-decomposable'

            param_dict['head_type'] = 'single'
            # param_dict['prediction_format'] = 'sparse'
            param_dict['default_rule'] = str(self.options.default_rule).lower()
            if self.options.default_rule:
                param_dict['max_rules'] += 1

            self.model = Boomer(**param_dict)

            # split data into train and test sets
            self.test_size = self.options.testsplit
            if (self.test_size > 0):
                self.X_train, self.X_test, self.Y_train, self.Y_test = \
                        train_test_split(self.X, self.Y, test_size=self.test_size,
                                random_state=self.seed)
            else:
                self.X_train = self.X
                self.X_test = [] # need a fix
                self.Y_train = self.Y
                self.Y_test = []# need a fix

            # check if we have info about categorical features
            if (self.use_categorical):
                self.categorical_features = from_data.categorical_features
                self.categorical_names = from_data.categorical_names
                self.target_name = from_data.class_names

                ####################################
                # this is a set of checks to make sure that we use the same as anchor encoding
                cat_names = sorted(self.categorical_names.keys())
                assert(cat_names == self.categorical_features)
                self.encoder = {}
                for i in self.categorical_features:
                    self.encoder.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                    self.encoder[i].fit(self.X[:,[i]])
            else:
                self.categorical_features = []
                self.categorical_names = []
                self.encoder = []

            fname = from_data
        elif from_model:
            fname = from_model
            self.load_datainfo(from_model)
            if (self.use_categorical is False) and (self.options.use_categorical is True):
                print("Error: Note that the model is trained without categorical features info. Please do not use -c option for predictions")
                exit()
        elif from_encoding:
            self.use_categorical = self.options.use_categorical
            fname = from_encoding

            # encoding, feature names, and number of classes
            # are read from an input file
            if fname.endswith('.cnf'):
                enc = MXEncoderBR(None, [], None, self, from_encoding)
                self.enc, self.intvs, self.imaps, self.ivars, self.feature_names, \
                        self.num_class = enc.access()
                self.mxe = enc
                
        # prepare the extra file names
        try:
            os.stat(options.output)
        except:
            os.mkdir(options.output)

        self.mapping_features()

        bench_name = os.path.splitext(os.path.basename(options.files[0]))[0]
        bench_dir_name = options.output + "/" + bench_name
        try:
            os.stat(bench_dir_name)
        except:
            os.mkdir(bench_dir_name)
        
        self.basename = (os.path.join(bench_dir_name, bench_name +
                        "_maxrule_" + str(options.n_estimators) +
                        "_maxcond_" + str(options.maxdepth) +
                        "_testsplit_" + str(options.testsplit)))
        # 251119: add the basename with possible weight digit limit
        if not options.w_digits is None:
            self.basename += '_wd_' + str(options.w_digits)

        self.modfile = self.basename + '.mod.pkl'
        self.mod_plainfile =  self.basename + '.mod.txt'
        self.resfile =  self.basename + '.res.txt'
        self.encfile =  self.basename + '.enc.txt'
        self.expfile =  self.basename + '.exp.txt'

    ###################################
    #### data_saving operations #######
    ###################################
    def form_datefile_name(self, modfile):
        data_suffix =  '.splitdata.pkl'
        return  modfile + data_suffix
    
    def pickle_save_file(self, filename, data):
        try:
            f =  open(filename, "wb")
            pickle.dump(data, f)
            f.close()
        except:
            print("Cannot save to file", filename)
            exit()
    
    def pickle_load_file(self, filename):
        try:
            f =  open(filename, "rb")
            data = pickle.load(f)
            f.close()
            return data
        except:
            print("Cannot load from file", filename)
            exit()

    def save_datainfo(self, filename):
        # print("saving  model to ", filename)
        self.pickle_save_file(filename, self.model)

        filename_data = self.form_datefile_name(filename)
        # print("saving  data to ", filename_data)
        samples = {}
        samples["X"] = self.X
        samples["Y"] = self.Y
        samples["X_train"] = self.X_train
        samples["Y_train"] = self.Y_train
        samples["X_test"] = self.X_test
        samples["Y_test"] = self.Y_test
        samples["feature_names"] = self.feature_names
        samples["target_name"] = self.target_name
        samples["num_class"] = self.num_class
        samples["categorical_features"] = self.categorical_features
        samples["categorical_names"] = self.categorical_names
        samples["encoder"] = self.encoder
        samples["use_categorical"] = self.use_categorical
        self.pickle_save_file(filename_data, samples)

    def load_datainfo(self, filename):
        print("loading model from ", filename)
        self.model = Boomer()
        self.model = self.pickle_load_file(filename)

        datafile = self.form_datefile_name(filename)
        print("loading data from ", datafile)
        loaded_data = self.pickle_load_file(datafile)
        # print('data is loaded!!')
        self.X = loaded_data["X"]
        self.Y = loaded_data["Y"]
        self.X_train = loaded_data["X_train"]
        self.X_test = loaded_data["X_test"]
        self.Y_train = loaded_data["Y_train"]
        self.Y_test = loaded_data["Y_test"]
        self.feature_names = loaded_data["feature_names"]
        self.target_name = loaded_data["target_name"]
        self.num_class = loaded_data["num_class"]
        self.nb_features = len(self.feature_names)
        self.categorical_features = loaded_data["categorical_features"]
        self.categorical_names = loaded_data["categorical_names"]
        self.encoder = loaded_data["encoder"]
        self.use_categorical = loaded_data["use_categorical"]

    ###################################
    #### Sample transforming operations #######
    ###################################
    def transform_x_cases(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.encoder != [])
            tx = []
            for i in range(self.nb_features):
                self.encoder[i].drop = None
                if (i in self.categorical_features):
                    tx_aux = self.encoder[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_y_labels(self, y):
        '''
            Boomer orginally deals with multi-label classification (a vector of binary values indicating the presence of labels)
            For multi-class classification, it needs to transform the multi-class value into the multi-label format
        '''
        # pass
        if self.num_class > 2:
            Y_transform = []
            # Only for multi-classification case
            for l in y:
                l_transform = [0] * self.num_class
                l_transform[l] = 1
                Y_transform.append(l_transform)
            return Y_transform
        else:
            return y

    def transform_inverse_y_labels(self, y_predict):
        '''
            Inversely transform the multi-label vector of y to a single predicted value
            Assuming that the y_lables corresponds to an exact class, and the class ransk from [0, len(y_label)-1]
        '''
        if self.num_class > 2:
            Y_inverse_transform = []
            # Only for multi-classificaation case
            for vec in y_predict:
                assert len(vec) == self.num_class
                assert sum(vec) == 1
                Y_inverse_transform.append(np.where(vec == 1)[0][0])
            return Y_inverse_transform
        else:
            return y_predict

    def mapping_features(self):
        """
            Same as XGBooster.py
            mapping the original features names with standard f_index
            considering the usage of feature encoding for categorical features
        """
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.encoder[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)

    ###################################
    #### Training/Encoding/Explaining #######
    ###################################
    def _format_rules_extracted(self):
        '''
            2025-7-1: Boomer version 0.12.0
            Return the formatted str of rules extracted
        '''
        assert isinstance(self.model, Boomer)
        assert not self.model.model_ is None

        rule_cnt = 0
        rule_str = ''
        for rule in self.model.model_:
            rule_str += 'rule[' + str(rule_cnt) + ']:\n'
            rule_cnt += 1
            # extract the body (conjunction of conditions)
            rule_str += '{'
            for cond in rule.body:
                rule_str += '[' + self.data.names[cond.feature_index] + cond.comparator + str(cond.threshold) + ']'
            rule_str += '} -> {'

            # extract the head (predictions of each label (originally multi-label classification)) 
            for pred in rule.head:
                rule_str += '[' + str(pred.output_index) + '=' + str(pred.value) + ']'
            rule_str += '}\n'
        return rule_str

    def train(self, outfile=None):
        """
            Train a Boosted Rule model using Boomer
        """
        return self.build_boomer_rules(outfile)
    
    def _extract_num_unique_conds_rule(self):
        unique_conds_rules = []
        for rule in self.model.model_:
            rule_conds = []
            for cond in rule.body:
                rule_conds.append('[' + str(cond.feature_index) + cond.comparator + str(cond.threshold) + ']')
            rule_conds_str = ''.join(sorted(rule_conds))

            if not rule_conds_str in unique_conds_rules:
                unique_conds_rules.append(rule_conds_str)
        return len(unique_conds_rules)

    def _postprocess_BR_weight_digit(self):
        '''
            251117: the function of realising the post-processing for limiting the weight digits for the Boosted Rules learned.
            calling this function the options.w_digits is asserted as a valid integer.
        '''
        assert not self.options.w_digits is None and isinstance(self.options.w_digits, int)
        assert isinstance(self.model, Boomer)
        assert not self.model.model_ is None
        
        stat = self.model.model_.__reduce__()
        n_stat = copy.deepcopy(stat)

        for i in range(len(stat[2][1][0])):
            lst = list(stat[2][1][0][i][-1])
            lst[0] = np.round(stat[2][1][0][i][-1][0], self.options.w_digits)
            n_stat[2][1][0][i][-1] = tuple(lst)

        self.model.model_.__setstate__(n_stat[-1])
        # print(self._format_rules_extracted())

    def build_boomer_rules(self, outfile=None):
        """
            Build an ensemble of rules
        """
        if outfile is None:
            outfile = self.modfile
        else:
           self.datafile = self.form_datefile_name(outfile)

        # fit the model with the training data
        # self.model.fit(self.transform_x_cases(self.X_train), self.Y_train)
        self.model.fit(self.transform_x_cases(self.X_train), self.transform_y_labels(self.Y_train))

        # 251118: add the post-processing to limit the weight digits
        if not self.options.w_digits is None:
            self._postprocess_BR_weight_digit()

        ### Saving data infos
        self.save_datainfo(outfile)

        # 2025-7-1: save the information of rules 
        # print("Saving plain models to ", self.mod_plainfile)
        with open(self.mod_plainfile, 'w') as f:
            f.write(self._format_rules_extracted())

        # need evaluate the model performance
        Y_train_predict = self.transform_inverse_y_labels(self.model.predict(self.transform_x_cases(self.X_train)))
        train_acc = round(accuracy_score(self.Y_train, Y_train_predict), 3)

        try:
            Y_test_predict = self.transform_inverse_y_labels(self.model.predict(self.transform_x_cases(self.X_test)))
            test_acc = round(accuracy_score(self.Y_test, Y_test_predict), 3)
        except:
            print("No results test data")
            test_acc = 0

        # Saving the results:
        with open(self.resfile, 'w') as f:
            f.write("{} & {} & {} &{}  &{}  \\\\ \n \\hline \n".format(
                                           os.path.basename(self.options.files[0]).replace("_","-"),
                                           train_acc,
                                           test_acc,
                                           self.options.n_estimators,
                                           self.options.testsplit))

        # 251022: fix the number of unique conds_rule
        return train_acc, test_acc, self.mod_plainfile, self._extract_num_unique_conds_rule()

    def _get_num_cond_pred_rule_body(self, rule):
        '''
            return the number of conditions and predictions in the head for a given rule
        '''
        cnt_cond, cnt_pred = 0, 0
        for cond in rule.body:
            cnt_cond += 1
        for pred in rule.head:
            cnt_pred += 1
        return cnt_cond, cnt_pred
    
    def predict_scores(self, sample):
        # compute the scores of all classes, to determine the predicted class
        csum = [[] for c in range(self.num_class)]
        sample_transformed = list(self.transform_x_cases(sample)[0])

        # traverse the boosted rules
        # rule_cnt = 0
        for rule in self.model.model_:
            # rule_cnt += 1
            cnt_cond_sat = 0
            for cond in rule.body:
                fval = cond.threshold
                leq = True if cond.comparator == '<=' else False
                sample_val = sample_transformed[cond.feature_index]
                assert (not sample_val is None)
                # 251222: BUG found => python can not accurately making the equality checking => fixed
                if (leq and ((sample_val < fval) or math.isclose(sample_val, fval, abs_tol=1e-6))) or ((not leq) and (sample_val > fval)):
                    cnt_cond_sat += 1
                else:
                    break
            
            cnt_cond, _ = self._get_num_cond_pred_rule_body(rule)
            if cnt_cond_sat == cnt_cond:
                # The rule is satisifed, count the pred_score in head
                for pred in rule.head:
                    csum[pred.output_index].append(pred.value)
                    # print(csum[1])
                    # 251007: consider the special case of binary classification
                    if self.num_class == 2:
                        csum[(pred.output_index + 1) % 2].append(-pred.value)

        return [sum(scores) for scores in csum]

    def encode(self, test_on=None):
        """
            Encode the boosted rules trained previously
        """
        assert self.options.encode in ['mx', 'mxe', 'maxsat', 'mxint', 'mxa', 'sat']
        encoder = MXEncoderBR(self.model, self.feature_names, self.num_class, self)
        self.mxe = encoder
        
        binary = True if self.num_class == 2 else False
        self.enc, self.intvs, self.imaps, self.ivars = encoder.encode(binary=binary)
        
        if test_on:
            encoder.test_sample(np.array(test_on))
            # print('log: Test on passed!')
        
        encoder.save_to(self.encfile)

        # 251219: preapare the explainer in the encode function
        assert self.options.encode in ['mx', 'mxe', 'maxsat', 'mxint', 'mxa', 'sat']
        # 251209: adding the option of SAT approach
        if self.options.encode == 'sat':
            self.x = SATExplainerBR(self.enc, self.intvs, self.imaps, self.ivars, self.feature_names,
                    self.num_class, self.options, self)
            # print('log: SATExplainerBR created!!')
        else:
            self.x = MXExplainerBR(self.enc, self.intvs, self.imaps, self.ivars,
                    self.feature_names, self.num_class, self.options, self)

    def explain(self, sample, expl_ext=None, prefer_ext=False):
        """
            Explain a prediction made for a given sample by the boosted rules
        """
        if self.options.encode == 'sat':
            expl = self.x.explain(np.array(sample))
        else:
            expl = self.x.explain(np.array(sample), self.options.smallest, expl_ext, prefer_ext)

        return expl
