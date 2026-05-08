#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## encode.py => containing the MaxSAT encoding class for Boosted Rules
# Date: 2025-07-31
# Author: Hao Hu

#
#==============================================================================
import six
import numpy as np
import collections
from decimal import Decimal
import time
import math

from pysat.formula import IDPool, CNF
from pysat.process import Processor

from .reason import MXReasonerBR, SATReasonerBR

# a named tuple for class encodings
#==============================================================================
ClassEnc = collections.namedtuple('ClassEnc', ['formula', 'heads', 'ivars'])
#
#==============================================================================
class MXEncoderBR(object):
    """
        MaxSAT encoder reasoner for Boosted Rules
    """
    def __init__(self, model, feats, nof_classes, boomer, from_file=None):
        
        self.model = model
        self.feats = {f: i for i, f in enumerate(feats)}
        self.nofcl = nof_classes
        self.idmgr = IDPool()
        self.options = boomer.options
        
        self.boomer = boomer

        # variable to feature ID
        self.vid2fid = {}
        # intvs => intervals;
        # imaps => interval index mapping
        # ivars => interval bool vars (l_{j,k})
        # lvars => lower bound bool vars (o_{j,k})
        self.intvs, self.imaps, self.ivars, self.lvars = None, None, None, None

        if from_file:
            self.load_from(from_file)

    def load_from(self, infile):
        """
            Loads the encoding from an input file => possibly reuse the encoding file
            => same as XGBooster.encode.load_from()
        """
        self.enc = CNF(from_file=infile)

        # empty intervals for the standard encoding
        self.intvs, self.imaps, self.ivars = {}, {}, {}

        for line in self.enc.comments:
            if line.startswith('c i ') and 'none' not in line:
                f, arr = line[4:].strip().split(': ', 1)
                f = f.replace('-', '_')
                self.intvs[f], self.imaps[f], self.ivars[f] = [], {}, []

                for i, pair in enumerate(arr.split(', ')):
                    ub, symb = pair.split(' <-> ')
                    ub = ub.strip('"')
                    symb = symb.strip('"')

                    if ub[0] != '+':
                        ub = float(ub)

                    self.intvs[f].append(ub)
                    self.ivars[f].append(symb)
                    self.imaps[f][ub] = i

            elif line.startswith('c features:'):
                self.feats = line[11:].strip().split(', ')
            elif line.startswith('c classes:'):
                self.nofcl = int(line[10:].strip())

    def compute_intervals(self):
        """
            compute all intervals used in Boosted rules + encode the domain relationship
            Unlike Trees of xgboost in JSON format, Boosted rules are in {[condition1]...} -> {[labelweight]} format
            The comparator should be considered in encoding the rules
        """
        # initialize the intervals dict {'f0':[], etc}
        self.intvs = {'{0}'.format(i): set([]) for i in self.boomer.extended_feature_names_as_array_strings}

        # get all thresholds of all features
        for rule in self.model.model_:
            for cond in rule.body:
                f = self.boomer.extended_feature_names_as_array_strings[cond.feature_index]
                v = cond.threshold
                self.intvs[f].add(v)
        
        # fliter the features not apprear in the rules and sort the intervals
        self.intvs = dict(filter(lambda x: len(x[1]) != 0, self.intvs.items()))
        self.intvs = {f: sorted(self.intvs[f]) + ['+'] for f in six.iterkeys(self.intvs)}

        #### encoding the valid feature domain relationship
        self.imaps, self.ivars, self.lvars = {}, {}, {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {intvs[i]: i for i in range(len(intvs))} # => {'f0': {intv0: 0, intv1: 1...}, ..}
            self.ivars[feat] = [None for i in range(len(intvs))]
            self.lvars[feat] = [None for i in range(len(intvs))]

            # The special case for the first interval
            # l_{j,1} <-> o_{j,1} ==> share the same var_id
            self.lvars[feat][0] = self.idmgr.id('{0}_lvar{1}'.format(feat, 0))
            self.ivars[feat][0] = self.lvars[feat][0]

            # The common part of the order + domain encoding
            for i in range(1, len(intvs) - 1):
                lvar_i = self.idmgr.id('{0}_lvar{1}'.format(feat, i))
                ivar_i = self.idmgr.id('{0}_ivar{1}'.format(feat, i))
                lvar_prev = self.lvars[feat][i - 1]

                # ordering encoding (o_{j, k} -> o_{j, k+1})
                self.enc['common'].append([-lvar_prev, lvar_i])

                # domain encoding (l_{j, k+1} <-> (\neg o_{j, k} /\ o_{j, k+1}))
                self.enc['common'].append([-ivar_i, -lvar_prev])
                self.enc['common'].append([-ivar_i, lvar_i])
                self.enc['common'].append([lvar_prev, -lvar_i, ivar_i])

                # saving bool vars
                self.lvars[feat][i] = lvar_i
                self.ivars[feat][i] = ivar_i
            
            # The special case of the last interval ('+' ==> '+inf')
            # l_{j, m_j +1} <-> \neg o_{j, m_j} ==> sharing same var_id
            self.lvars[feat][-1] = -self.lvars[feat][-2] # o_{j, m_j + 1} is never used to encode => just used to note l_{j, m_j +1}
            self.ivars[feat][-1] = self.lvars[feat][-1]

            # There is atleast one interval being selected
            if len(intvs) > 2:
                self.enc['common'].append(self.ivars[feat])
            
        # mapping bool var_ids to feature id
        for feat in self.ivars:
            for v, ub in zip(self.ivars[feat], self.intvs[feat]):
                self.vid2fid[v] = (feat, ub)

    def traverse_rules(self, binary=False):
        """
            traverse all rules in the boosted rules model + encode paths + rule head
            We don't need provide the class id as the corresponded rules can be separated directly.
            251007: #BUG found for binary classification
            #Prob: for binary case, the rules just have heads with weights in predicted class (0 / 1), not others
            #Solve: add the clauses and weights for the other class with negative weights
        """
        # print('-- Starting traverse rules!!')
        for rule in self.model.model_:
            # encode the rule body
            path_lvars, cond_ivars = [], []
            for cond in rule.body:
                feat = self.boomer.extended_feature_names_as_array_strings[cond.feature_index]
                fval = cond.threshold
                comparator = 1 if cond.comparator == '<=' else -1    
                # get the corresponding bool var
                intv_index = self.imaps[feat][fval]
                path_lvars.append(self.lvars[feat][intv_index] * comparator)
                
                # 251020: all ivars of the feat as into cond_ivars (gurantee the positive) => for future freeze 
                cond_ivars += [abs(ivar) for ivar in self.ivars[feat]]

            head_var = self.idmgr.id(tuple(sorted(path_lvars)))
            if path_lvars:
                # print('log: before head_var checking! {}'.format(self.enc['paths']))
                # t = time.time()
                if not head_var in self.enc['paths']:
                    # encode h_i -> /\ o_j_k
                    for var in path_lvars:
                        self.enc['paths'][head_var].append([var, -head_var])
                    # enode /\ o_j_k -> h_i
                    self.enc['paths'][head_var].append([-var for var in path_lvars] + [head_var])
                # print('log: after head_var checking {}!'.format(time.time() - t))
                
                cnt_pred = 0
                for pred in rule.head:
                    cnt_pred += 1
                    # copy the encoding into current class encoding
                    self.enc[pred.output_index].formula.extend(self.enc['paths'][head_var])
                    self.enc[pred.output_index].heads.append((head_var, pred.value))
                    # 251020: copy the cond_ivars to the predicted class
                    # for ivar in cond_ivars:
                        # 251204: NOTE: the judgement is so time costing!! Need be improved to better effiency in encoding
                        # if not ivar in self.enc[pred.output_index].ivars:
                        # self.enc[pred.output_index].ivars.append(ivar)
                    # using the set to remove duplicate ivars 
                    self.enc[pred.output_index].ivars.extend(cond_ivars)

                    # binary case: add the reverse weights for the other class
                    if binary:
                        reverse_class = (pred.output_index + 1) % 2
                        self.enc[reverse_class].formula.extend(self.enc['paths'][head_var])
                        self.enc[reverse_class].heads.append((head_var, -pred.value))
                        # 251020: copy the cond_ivars to the predicted class
                        # for ivar in cond_ivars:
                        #     if not ivar in self.enc[reverse_class].ivars:
                        #         self.enc[reverse_class].ivars.append(ivar)
                        self.enc[reverse_class].ivars.extend(cond_ivars)

                # single-output BOOMER, which predict exact one label (except the deafult one)
                assert cnt_pred == 1
            else:
                # the deafult rule without the conditions => with all labels
                for pred in rule.head:
                    self.enc[pred.output_index].formula.append([head_var])
                    self.enc[pred.output_index].heads.append((head_var, pred.value))
                    # for ivar in cond_ivars:
                    #     if not ivar in self.enc[pred.output_index].ivars:
                    #         self.enc[pred.output_index].ivars.append(ivar)
                    self.enc[pred.output_index].ivars.extend(cond_ivars)

                    # binary case: add the reverse weights for the other class
                    if binary:
                        reverse_class = (pred.output_index + 1) % 2
                        self.enc[reverse_class].formula.append([head_var])
                        self.enc[reverse_class].heads.append((head_var, -pred.value))
                        # 251020: copy the cond_ivars to the predicted class
                        # for ivar in cond_ivars:
                        #     if not ivar in self.enc[reverse_class].ivars:
                        #         self.enc[reverse_class].ivars.append(ivar)
                        self.enc[reverse_class].ivars.extend(cond_ivars)
        # print('-- Rules traversed!!')

    def make_varpos(self):
        """
            Traverse all the vars and get their positions in the list of inputs.
            => self.vpos shows the bool_var_id corresponding to which index of the feature
        """
        self.vpos, pos = {}, 0

        for feat in self.ivars:
            if '_' in feat or len(self.ivars[feat]) == 2:
                for lit in self.ivars[feat]:
                    if abs(lit) not in self.vpos:
                        self.vpos[abs(lit)] = pos
                        pos += 1
            else:
                for lit in self.ivars[feat]:
                    if abs(lit) not in self.vpos:
                        self.vpos[abs(lit)] = pos
                pos += 1

    def encode(self, binary=False):
        """
            The overall function for realising the MaxSAT encoding
        """ 
        # initial encoding
        self.enc = {}

        # we can create the classEnc for each class but only use the label corresponding one
        for j in range(self.nofcl):
            self.enc[j] = ClassEnc(formula=CNF(), heads=[], ivars=[])

        # common clauses 
        self.enc['common'] = []
        # rules (paths) encodings
        self.enc['paths'] = collections.defaultdict(lambda: [])

        # (Hard - Interval) Computing all intervals and encode the domain relationship
        # Just dealing in 'common'
        self.compute_intervals()

        # (Hard - Condition + Head)
        # Dealing in 'path' and enc[j]
        self.traverse_rules(binary=binary)

        self.make_varpos()

        # considering the categorical feature special case
        categories = collections.defaultdict(lambda: [])
        expected = collections.defaultdict(lambda: 0)
        for f in self.boomer.extended_feature_names_as_array_strings:
            if '_' in f:
                if f in self.ivars:
                    categories[f.split('_')[0]].append(self.ivars[f][1])
                expected[f.split('_')[0]] += 1
        for c, feats in six.iteritems(categories):
            if len(feats) > 1:
                if len(feats) == expected[c]:
                    self.enc['common'].extend(CardEnc.equals(feats,
                        vpool=self.idmgr, encoding=self.optns.cardenc))
                else:
                    self.enc['common'].extend(CardEnc.atmost(feats,
                        vpool=self.idmgr, encoding=self.optns.cardenc))        
        
        # number of clauses considering the only single class to be explained
        # nof_clauses_singleclass = sum([len(self.enc[j].formula.clauses) for j in range(self.nofcl)]) + len(self.enc['common'])

        # extend all class formulas with the hard clauses in 'common'
        for j in range(self.nofcl):
            self.enc[j].formula.extend(self.enc['common'])
            self.enc[j].formula.nv = self.idmgr.top

        # nb of clauses (all class considered) and bool vars (all class considered)
        # NOTE: as we consider only the final associated class of the test_sample,
        #   therefore, the exact nb of clauses and bool vars would be reduced when given the instance to explain.
        nof_clauses = sum([len(self.enc[j].formula.clauses) for j in range(self.nofcl)])
        nof_bool_vars = self.idmgr.top

        self.nof_bool_vars = nof_bool_vars
        self.nof_clauses = nof_clauses
        # self.nof_clauses_single = nof_clauses_singleclass

        if self.options.verb:
            print("bool vars (in total): " + str(nof_bool_vars))
            print("clauses (in total): {0}".format(nof_clauses))
            print("clauses for common domain: {}".format( len(self.enc['common'])))
            for j in range(self.nofcl):
                print("clauses for class {0}: {1}-paths".format(j, len(self.enc[j].formula.clauses) - len(self.enc['common'])))
            # print("encoding clauses (for single class considered): {0}".format(nof_clauses_singleclass))

        # delete unnecessary clasuses
        del self.enc['common']
        del self.enc['paths']

        return self.enc, self.intvs, self.imaps, self.ivars
    
    def get_literals(self, sample):
        """
            Get the corresponding literals of a sample given
        """
        lits = []

        sample_internal = list(self.boomer.transform_x_cases(sample)[0])
        # print('log: sample_internal is {}'.format(sample_internal))
        for feat, fval in zip(self.boomer.extended_feature_names_as_array_strings, sample_internal):
            if feat in self.intvs:
                # determin the corresponding intervals
                for ub, fvar in zip(self.intvs[feat], self.ivars[feat]):
                    # 251222: #BUG found, fval <= ub is not correctly checked as float for python
                    if ub == '+' or (fval < ub or math.isclose(fval, ub, abs_tol=1e-6)):
                        lits.append(fvar)
                        break
        return lits

    def test_sample(self, sample):
        """
            Check whether the encoding "predicts" the same class as the boosted rules done
            for given input sample
        """
        # final pred_score for each class
        cscores = self.boomer.predict_scores(sample)
        escores = None
        if self.options.relax:     # relax the demical point (0 in default)
            cscores = [round(score, self.options.relax) for score in cscores]
        cwinner = cscores.index(max(cscores))
        
        # Obtaining the scores computed via the MaxSAT encoding
        hypos = self.get_literals(sample)
        if self.options.verb:
            # print("Testing sample is: ", list(sample))
            print("log: With the predicted class " + str(cwinner)) 
            # print('log: The hypos are {}'.format(hypos))

        # The deafult is using the internal solver (int)
        if self.options.encode == 'mxa':
            ortype = 'alien'
        elif self.options.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'

        print('Boomer scores are: ', [float(str(cs)) for cs in cscores])
        # 251201: Calling the SATReasonerBR
        if self.options.encode == 'sat':
            with SATReasonerBR(self.enc, cwinner, solver=self.options.solver, 
                            pb_encoding=self.options.pb_encoding, wd=self.options.w_digits, 
                            preprocess=self.options.formula_preprocess) as x:
                assert x.get_coex(hypos) == None, 'Wrong class predicted by the SAT encoding'
                print('SAT encoding passed!!')
        else:
            # Calling Reasoner => K-1 MaxSAT oracle
            with MXReasonerBR(self.enc, cwinner, solver=self.options.solver,
                    oracle=ortype) as x:
                assert x.get_coex(hypos) == None, 'Wrong class predicted by the MaxSAT encoding'
                escores = x.get_scores()
            # print([abs(Decimal(cscores[i]) - Decimal(escores[i])) for i in range(len(cscores))])
            assert all(map(lambda c, e: abs(Decimal(c) - Decimal(e)) <= Decimal(0.001), cscores, escores)), \
                    'wrong prediction: {0} vs {1}'.format(cscores, escores)

        if self.options.verb:
            print('Boomer scores are: ', [float(str(cs)) for cs in cscores])
            if escores is not None:
                print('MaxSAT scores are: ', [float(str(e)) for e in escores])
            # print('Original nb of hypos = {}, hypos = {}'.format(len(hypos), hypos))

    def save_to(self, outfile):
        """
            NOTE: 8-28 stroing all classes clauses (possibly change in future for the predicted class)
            Saving the MaxSAT encoding into an output file 
        """
        if outfile.endswith('.txt'):
            outfile = outfile[:-3] + 'cnf'

        formula = CNF()
        # comments
        formula.comments = ['c features: {0}'.format(', '.join(self.feats)),
                'c classes: {0}'.format(self.nofcl)]

        for clid in self.enc:
            formula.comments += ['c clid starts: {0} {1}'.format(clid, len(formula.clauses))]
            for head in self.enc[clid].heads:
                formula.comments += ['c leaf: {0} {1} {2}'.format(clid, *head)]
            formula.clauses.extend(self.enc[clid].formula.clauses)

        for f in self.boomer.extended_feature_names_as_array_strings:
            if f in self.intvs:
                c = 'c i {0}: '.format(f)
                c += ', '.join(['"{0}" <-> "{1}"'.format(u, v) for u, v in zip(self.intvs[f], self.ivars[f])])
            else:
                c = 'c i {0}: none'.format(f)

            formula.comments.append(c)
        
        formula.to_file(outfile)

                