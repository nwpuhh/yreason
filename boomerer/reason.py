#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
##
##  Created on: Aug 26
##      Change from mxreason.py to satisify the BOOMER
##      Author: Hao Hu
##

# imported modules:
#==============================================================================
from __future__ import print_function
import collections
import copy

from .erc2 import ERC2
from pysat.examples.rc2 import RC2Stratified
from pysat.formula import CNF, CNFPlus, WCNF, IDPool
from pysat.process import Processor
from pysat.pb import PBEnc, EncType
from pysat.solvers import Solver as SATSolver

import decimal
from functools import reduce
import math
import subprocess
import sys
import tempfile
import gc

import time

#
#==============================================================================
class SATReasonerBR:
    """
        SAT-based explanation oracle. It contains the corresponding PB constraints
        to decide whether a given set of feature values would change the original prediction.
    """
    def __init__(self, encoding, target, solver='g3', verbose=0, pb_encoding=0, wd=None, preprocess=True):
        '''
            Initialiser
        '''
        print('log: reasoner: init start')
        self.oracles = {}   # SAT oracles 
        self.formulas = {}  # CNF formulas for each oracle
        self.values = collections.defaultdict(lambda: [])  # lits, values for all the classes
        self.size_formula = {} # item of format (#var, #clauses_orig, #clauses_processed)
        
        # preprocessing stuff 
        self.preprocess = preprocess
        self.preprocessors = {}
        self.freeze_vars = {}

        # SAT oracle options
        self.pb_encoding = pb_encoding
        self.solver = solver

        self.verbose = verbose
        self.target = target
        # wd measures the weight digits used to encode the PB constraint
        # wd=None => default value, preseving 3 digits; otherwise, directly the value given
        if wd is None:
            self.wd = 3
        else:
            assert isinstance(wd, int)
            self.wd = wd

        # the real initialization
        self.init(encoding, solver)

        print('log: reasoner: init end')

    def __del__(self):
        """
            Magic destructor.
        """
        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """
        self.delete()

    def delete(self):
        '''
            actual deconstructor
        '''
        print('log: reasoner, __del__ start')
        if self.oracles:
            for oracle in self.oracles.values():
                if oracle:
                    oracle.delete()
            self.oracles.clear()
            self.oracles = None
            self.target = None

        if self.preprocessors:
            for processor in self.preprocessors.values():
                processor.delete()
            self.freeze_vars.clear()
            self.freeze_vars = None
        
        self.values.clear()
        self.values = None
        self.formulas.clear()
        self.formulas = None
        gc.collect()
        print('log: reasoner, __del__ end')
        
    ###### Initialization ######
    def init(self, encoding, solver):
        '''
            this function creates (nof_class - 1) SAT oracles
            Each one compare the target class with the current class (sum_{i,l})
            The SAT oracle encodes the Pseudo-Boolean constraints + preprocessing
        '''
        # preparing freeze_vars with head variables
        for clid in encoding:
            self.freeze_vars[clid] = []
            for lit, wght in encoding[clid].heads:
                self.values[clid].append(tuple([lit, wght]))
                # the head_var of each clid need be freezed
                self.freeze_vars[clid].append(lit)

        # creatingg the formulas and SAT oracles
        for clid in encoding:
            # no need to compare the target
            if clid == self.target:
                continue
            
            # update the freeze_vars corresponding to self.target
            self.freeze_vars[clid] += self.freeze_vars[self.target]

            # Step 1: preparing domain ordering constraints
            self.formulas[clid] = CNFPlus()
            # self.formulas[clid] = CNF()
            for c in encoding[clid].formula:
                self.formulas[clid].append(c)
            # self.formulas[clid].extend(encoding[clid].formula)

            # preparing freeze_vars with interval variables
            self.freeze_vars[clid] += list(set(encoding[clid].ivars))
            if len(encoding) > 2:
                self.formulas[clid].extend(encoding[self.target].formula)
                self.freeze_vars[clid] += encoding[self.target].ivars
            
            # print('log: SATReasoner, before init_PB #clauses = {}'.format(len(self.formulas[clid].clauses)))

            # Step 2: preparing the PB constraints
            self.init_PB_constraints(encoding, clid)

            # 251209: storing the real # vars, #clauses
            clause_orig = len(self.formulas[clid].clauses)
            self.size_formula[clid] = (self.formulas[clid].nv, clause_orig)

            # Step 3: make the preprocessing if used
            if self.preprocess:
                # print('log: SATReasoner, clid={}, before #clauses = {}'.format(clid, len(self.formulas[clid].clauses)))
                self.preprocessors[clid] = Processor(bootstrap_with=self.formulas[clid])
                self.formulas[clid] = self.preprocessors[clid].process(freeze=self.freeze_vars[clid])
                self.size_formula[clid] = (self.formulas[clid].nv, clause_orig, len(self.formulas[clid].clauses))
                # print('after #clauses = {}'.format(len(self.formulas[clid].clauses)))

            # Step 4: create the SAT oracle via solver
            self.oracles[clid] = SATSolver(name=solver, bootstrap_with=self.formulas[clid])

    def _process_oppo_literals(self, wghts):
        '''
            processing the opposite literals => the solver accepts positive weights
            P1: treating the neative literals
            P2: flipping the id of literals into negative if they have negative weights
        '''
        # processing the opposite literals, if any.
        i, lits = 0, sorted(wghts.keys(), key=lambda l: 2 * abs(l) + (0 if l > 0 else 1))
        ##################################################
        while i < len(lits) - 1:
            if lits[i] == -lits[i + 1]:
                # if there are both l and \neg l
                l1, l2 = lits[i], lits[i + 1]
                minw = min(wghts[l1], wghts[l2], key=lambda w: abs(w))
                # updating the weights
                wghts[l1] -= minw
                wghts[l2] -= minw

                i += 2
            else:
                # not appliable for single version, as there will be no sharing 
                i += 1
        ##################################################
        # # flipping literals with negative weights
        # lits = list(wghts.keys())
        # for l in lits:
        #     if wghts[l] < 0:
        #         wghts[-l] = -wghts[l]
        #         del wghts[l]    
        # return wghts
        return wghts

    def init_PB_constraints(self, encoding, clid):
        '''
            preparing the PB constraints via transforming real weights into integer via multiplying 10^{wd}
        '''
        # Step 1: preparing the (lit, real_weight) list
        wghts= collections.defaultdict(lambda: 0)

        for label in (clid, self.target):
            if label != self.target:
                coeff = 1
            else:  # this is the target class
                if len(encoding) > 2:
                    coeff = -1
                else:
                    # LOG: no need to encode the binary classification case
                    # there are not 2*n_estimator as multi-classification, but only n_estimator
                    continue    
            # we can directly traverse the heads of each class
            for lit, wght in encoding[label].heads:        
                # the corresponding heads weight: w_clid - w_target
                wghts[lit] += coeff * wght

        # filtering out those with zero-weights => useless in the final prediction
        wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))
        # processing the oppsite literals conditions
        # wghts = self._process_oppo_literals(wghts)
        # wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))

        # Step 2: preparing the PB constraints, where atleast(lit * wght, bound=0)
        lits = list(wghts.keys())
        wghts_int = [int(round(wghts[l] * (10 ** self.wd))) for l in lits]
                
        gcd_wghts = reduce(math.gcd, wghts_int)
        wghts_int = [int(w / gcd_wghts) for w in wghts_int]

        # 251201: #BUG PBEnc Not Working!!! => fixed by not opposite the literal with negative weights
        # 251201: still need to add the top_id, to avoid problem of new createing variables in PB constraints
        # cnf_pb = PBEnc.atleast(lits=lits, weights=wghts_int, bound=0, encoding=self.pb_encoding, \
        #                         top_id=self.formulas[clid].nv)
        # self.formulas[clid].extend(cnf_pb.clauses)
        self.formulas[clid].extend(PBEnc.atleast(lits=lits, weights=wghts_int, bound=0, encoding=self.pb_encoding, \
                                top_id=self.formulas[clid].nv).clauses)
    ############################

    def get_coex(self, feats):
        '''
            A call to the oracle trying to obtain a counterexample of the given set of feature values.
            If such counterexample exist, return the model. Otherwise, return the None.

            Note that returning None indicating the given set of feature values is an WAXp (not necessarily a minimal one)
        '''
        # updating the reason
        self.reason = set()

        # solving the K-1 SAT oracles with assumptions
        for clid in self.oracles:
            if clid == self.target:
                continue
            # If the instance is wrongly predicted
            if self.oracles[clid].solve(assumptions=feats):
                if self.preprocess:
                    model = self.preprocessors[clid].restore(self.oracles[clid].get_model())
                else:
                    model = self.oracles[clid].get_model()
                # getting the score and return back the model
                # print('The original prediction {} is wrongly classified as {}'.format(self.target, clid))
                return model
            else:
                # cores[clid] = self.oracles[clid].get_core()
                # core_u = core_u.union(cores[clid])
                if not self.oracles[clid].get_core() is None:
                    self.reason = self.reason.union(self.oracles[clid].get_core())

        if not self.reason:
            self.reason = None

        # if no counterexample exist, return None
        # print('log: SATReasoner, cores = {}, len={}'.format(sorted(core_u, key=abs), len(core_u)))
        return None
    
    def get_reason(self, v2fmap=None):
        """
            Reports the last reason (analogous to unsatisfiable core in SAT).
            If the extra parameter is present, it acts as a mapping from
            variables to original categorical features, to be used a the
            reason.
        """

        assert self.reason, 'There no reason to return!'
        if v2fmap:
            return sorted(set(v2fmap[v] for v in self.reason))
        else:
            return self.reason

#
#==============================================================================
class MXReasonerBR:
    """
        MaxSAT-based explanation oracle. It can be called to decide whether a
        given set of feature values forbids any potential misclassifications,
        or there is a counterexample showing that the set of feature values is
        not an explanation for the prediction.
    """

    def __init__(self, encoding, target, solver='g3', oracle='int',
            am1=False, exhaust=False, minz=False, trim=0, verbose=0):
        """
            Magic initialiser.
        """
        self.verbose = verbose
        self.oracles = {}   # MaxSAT solvers => when single is triggered, just 1 valid oracle
        self.target = None  # index of the target class
        self.reason = None  # reason / unsatisfiable core
        self.values = collections.defaultdict(lambda: [])  # values for all the classes => single <-> 1 valid
        self.scores = {}  # class scores => single <-> 1 valid
        self.formulas = {}
        self.size_formula = {}

        # 251020: storing the preprocessors for each clid => reuse for further restore model
        self.processors = {}
        self.freeze_vars = {}
        # copying class values
        self.ortype = oracle

        # MaxSAT-oracle options
        self.am1 = am1
        self.exhaust = exhaust
        self.minz = minz
        self.trim = trim
        self.solver = solver  # keeping for alien solvers

        # doing actual initialisation
        self.target = target
        self.init_non_single(encoding, solver)

    def __del__(self):
        """
            Magic destructor.
        """

        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.delete()

    def init_non_single(self, encoding, solver):
        ''''
            For Non-singlg version, it creates (nof_class - 1) MaxSAT oracles
            Each of them compare the target and the current class. (sum_{i, l})
            251020: doing the preprocessing for the hard formula => 
                taking care to freeze the input Bool vars (lower bound), and the output Bool vars (heads)
        '''
        
        for clid in encoding:
            self.freeze_vars[clid] = []
            for lit, wght in encoding[clid].heads:
                self.values[clid].append(tuple([lit, wght]))
                # the head_var of each clid need be freezed
                self.freeze_vars[clid].append(lit)

        # creating the formulas and oracles
        for clid in encoding:
            if clid == self.target:
                continue
            
            # the freeze_vars also needs the head_vars of self.target
            self.freeze_vars[clid] += self.freeze_vars[self.target]

            # Step 1: adding hard clauses
            self.formulas[clid] = WCNF()
            
            hard_formula = CNF()
            hard_formula.extend(encoding[clid].formula)
            self.freeze_vars[clid] += list(set(encoding[clid].ivars))
            if len(encoding) > 2:
                # for cl in encoding[self.target].formula:
                #     if not cl in hard_formula.clauses:
                #         hard_formula.append(cl)
                hard_formula.extend(encoding[self.target].formula)
                self.freeze_vars[clid] += encoding[self.target].ivars
            # self.freeze_vars[clid] = list(set(self.freeze_vars[clid]))
            
            # 251020: adding the preprocessing for hard clauses!!
            self.processors[clid] = Processor(bootstrap_with=hard_formula)
            hard_formula_p = self.processors[clid].process(freeze=self.freeze_vars[clid])

            # WCNF extends with the hard_formula processed
            self.formulas[clid].extend(hard_formula_p)

            # Step 2: adding soft clauses and recording all the heads values
            self.init_soft_non_single(encoding, clid)

            # print('log: MXReasoner, clid={}, before #nb hard clauses={}, processed #nb hard clauses={}'.format(clid, len(hard_formula.clauses), len(hard_formula_p.clauses)))
            clause_soft = len(self.formulas[clid].soft)
            self.size_formula[clid] = (self.formulas[clid].nv, len(hard_formula.clauses) + clause_soft, len(hard_formula_p.clauses) + clause_soft)

            if self.ortype == 'int':
                # a new MaxSAT solver
                self.oracles[clid] = ERC2(self.formulas[clid], solver=solver,
                    adapt=self.am1, blo='cluster', exhaust=self.exhaust,
                    minz=self.minz, verbose=0)  

    def _process_oppo_literals_init_soft(self, wghts, cost):
        '''
            processing the opposite literals => the solver accepts positive weights
            P1: treating the neative literals
            P2: flipping the id of literals into negative if they have negative weights
        '''
        # processing the opposite literals, if any.
        i, lits = 0, sorted(wghts.keys(), key=lambda l: 2 * abs(l) + (0 if l > 0 else 1))
        ##################################################
        while i < len(lits) - 1:
            if lits[i] == -lits[i + 1]:
                l1, l2 = lits[i], lits[i + 1]
                minw = min(wghts[l1], wghts[l2], key=lambda w: abs(w))

                # updating the weights
                wghts[l1] -= minw
                wghts[l2] -= minw

                # updating the cost if there is a conflict between l and -l
                if wghts[l1] * wghts[l2] > 0:
                    cost += abs(minw)
                i += 2
            else:
                # not appliable for single version, as there will be no sharing 
                i += 1
        ##################################################

        # flipping literals with negative weights
        lits = list(wghts.keys())
        for l in lits:
            if wghts[l] < 0:
                cost += -wghts[l]
                wghts[-l] = -wghts[l]
                del wghts[l]
        
        return wghts, cost

    def init_soft_non_single(self, encoding, clid):
        """
            Non-single version, Processing the heads and creating the set of soft clauses.
            Need be changed: No need for the atmost1 constraints for each tree
        """
        wghts, cost = collections.defaultdict(lambda: 0), 0

        for label in (clid, self.target):
            if label != self.target:
                coeff = 1
            else:  # this is the target class
                if len(encoding) > 2:
                    coeff = -1
                else:
                    # LOG: no need to encode the binary classification case
                    # there are not 2*n_estimator as multi-classification, but only n_estimator
                    continue
            
            # we can directly traverse the heads of each class
            for lit, wght in encoding[label].heads:        
                # the corresponding heads weight: w_clid - w_target
                wghts[lit] += coeff * wght

        # filtering out those with zero-weights => useless in the final prediction
        wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))
    
        # processing the oppsite literals conditions
        wghts, cost = self._process_oppo_literals_init_soft(wghts, cost)

        # maximum value of the objective function
        self.formulas[clid].vmax = sum(wghts.values())

        # here is the start cost
        self.formulas[clid].cost = cost

        # adding remaining heads with non-zero weights as soft clauses
        for lit, wght in wghts.items():
            if wght != 0:
                self.formulas[clid].append([lit], weight=wght)

    def delete(self):
        """
            Actual destructor.
        """

        if self.oracles:
            for oracle in self.oracles.values():
                if oracle:
                    oracle.delete()
            
            # 251020: deleting all processors
            for proc in self.processors.values():
                proc.delete()

            self.oracles = {}
            self.target = None
            self.values = None
            self.reason = None
            self.scores = {}
            self.formulas = {}

    def get_coex(self, feats, full_instance=False, early_stop=False):
        """
            A call to the oracle to obtain a counterexample to a given set of
            feature values (may be a complete instance or a subset of its
            feature values). If such a counterexample exists, it is returned.
            Otherwise, the method returns None.

            Note that if None is returned, the given set of feature values is
            an abductive explanation for the prediction (not necessarily a
            minimal one).
        """
        # resetting the scores
        self.scores = {clid: 0 for clid in self.oracles}

        # updating the reason
        self.reason = set()

        if self.ortype == 'int':
            # using internal MaxSAT solver incrementally
            for clid in self.oracles:
                if clid == self.target:
                    continue

                model = self.oracles[clid].compute(feats, full_instance, early_stop)
                if not model is None:
                    # 251020: as the soft claues are all being frozen, therefore the processor in hard formula not affect the optimal solution of MaxSAT
                    # Then, just restore the model by the processor
                    model = self.processors[clid].restore(model)
                
                assert model or (early_stop and self.oracles[clid].cost > self.oracles[clid].slack), \
                        'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed to represent the predicted class!
                if model and (not self.target_winning(model, clid)):
                    return model

                # otherwise, proceed to another clid
                # 251104: try to avoid the union empty reason (of None)
                if not self.oracles[clid].get_reason() is None:
                    # print('log: clid={}, reason={}'.format(clid, self.oracles[clid].get_reason()))
                    self.reason = self.reason.union(set(self.oracles[clid].get_reason()))

            if not self.reason:
                self.reason = None

            # if no counterexample exists, return None
        else:
            # here we start an external MaxSAT solver every time
            for clid in self.formulas:
                if clid == self.target:
                    continue

                if self.ortype == 'ext':  # external RC2
                    with RC2Stratified(self.formulas[clid], solver='g3',
                            adapt=self.am1, blo='div', exhaust=self.exhaust,
                            incr=False, minz=self.minz, nohard=False,
                            trim=self.trim, verbose=0) as rc2:

                        # adding more hard clauses on top
                        for lit in feats:
                            rc2.add_clause([lit])

                        model = rc2.compute()
                else:  # expecting 'alien' here
                    # dumping the formula into a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wcnf') as fp:
                        sz = len(self.formulas[clid].hard)
                        self.formulas[clid].hard += [[l] for l in feats]
                        self.formulas[clid].to_file(fp.name)
                        self.formulas[clid].hard = self.formulas[clid].hard[:sz]
                        fp.flush()

                        outp = subprocess.check_output(self.solver.split() + [fp.name], shell=False)
                        outp = outp.decode(encoding='ascii').split('\n')

                    # going backwards in the log and extracting the model
                    for line in range(len(outp) - 1, -1, -1):
                        line = outp[line]
                        if line.startswith('v '):
                            model = [int(l) for l in line[2:].split()]

                assert model, 'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed
                # to represent the predicted class!
                if not self.target_winning(model, clid):
                    return model

            # otherwise, proceed to another clid
            self.reason = set(feats)

    def target_winning(self, model, clid):
        """
            Change as the target winning or not
        """
        for label in (self.target, clid):
            # computing the value for the current class label
            self.scores[label] = 0

            for lit, wght in self.values[label]:
                if model[abs(lit) - 1] > 0:
                    self.scores[label] += wght

        return True if self.scores[self.target] >= self.scores[clid] else False

    def get_scores(self):
        """
            Get all the actual scores for the classes computed with the previous call.
        """
        # this makes sense only for complete instances
        # BUG-251104: if there is a class sum of weight is 0.0, then it would be problem
        # assert all([score != 0 for score in self.scores.values()])

        return [self.scores[clid] for clid in sorted(self.scores.keys())]
        # return [self.scores[clid] for clid in range(len(self.scores))]

    def get_reason(self, v2fmap=None):
        """
            Reports the last reason (analogous to unsatisfiable core in SAT).
            If the extra parameter is present, it acts as a mapping from
            variables to original categorical features, to be used a the
            reason.
        """

        assert self.reason, 'There no reason to return!'

        if v2fmap:
            # print(sorted(self.reason, key=abs))
            # print(v2fmap)
            return sorted(set(v2fmap[v] for v in self.reason))
        else:
            return self.reason
