#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py for Boosted Rules
##
##  Created on: Sep 19, 2025
##      Author: Hao Hu 

#
#==============================================================================
from __future__ import print_function
import collections
from functools import reduce
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysat.solvers import Solver as SATSolver
import resource
from six.moves import range
import sys
# import gc, tracemalloc

from .reason import MXReasonerBR, SATReasonerBR

#
#==============================================================================
class SATExplainerBR(object):

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, boomer):
        """
            Constructor.
        """
        print('log: explainer: init start')
        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()
        self.fcats = []
        self.formula = formula

        # saving Boomerer
        self.boomer = boomer
        self.verbose = self.optns.verb

        # SAT-based oracles
        self.oracles = {}
        self.oracle = None

        # 260226: Memory problem: we apply option 2!!
        # Option 1: we can build K SATReasonBR objects, each one contains K-1 SAT oracles
        # => no need to rencode the SAT oracles when explaining instances with different predictions,
        # => but large in memory when K is large, like MNIST K=10, then it needs 90 SAT oracles in total.
        # Option 2: we can build only 1 SATReasonBR object containing K-1 SAT oracles, when the prediction is judged.
        # => reduce the memory, but needs rencode the SAT oracles multiple times.
        
        ## Option 1
        # for clid in range(nof_classes):
        #     self.oracles[clid] = SATReasonerBR(formula, clid, 
        #                 solver=self.optns.solver, pb_encoding=self.optns.pb_encoding, 
        #                 wd=self.optns.w_digits, preprocess=self.optns.formula_preprocess)

        # poracle => SAT oracle to make the prediction
        self.poracle = SATSolver(name='g3')
        for clid in range(nof_classes):
            self.poracle.append_formula(formula[clid].formula)

        # determining which features should go hand in hand
        categories = collections.defaultdict(lambda: [])
        for f in self.boomer.extended_feature_names_as_array_strings:
            if f in self.ivars:
                if '_' in f or len(self.ivars[f]) == 2:
                    categories[f.split('_')[0]].append(self.boomer.mxe.vpos[self.ivars[f][0]])
                else:
                    for v in self.ivars[f]:
                        # this has to be checked and updated
                        categories[f].append(self.boomer.mxe.vpos[abs(v)])

        # these are the result indices of features going together
        self.fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
        self.fcats_copy = self.fcats[:]

        # all used feature categories
        self.allcats = list(range(len(self.fcats)))

        # variable to original feature index in the sample
        self.v2feat = {}
        for var in self.boomer.mxe.vid2fid:
            feat, ub = self.boomer.mxe.vid2fid[var]
            self.v2feat[var] = int(feat.split('_')[0][1:])

        # number of oracle calls involved
        self.calls = 0

        print('log: explainer: init end')

    def __del__(self):
        self.delete()
        
    def delete(self):
        '''
            actual deconstructor
        '''
        if self.oracles:
            for clid, oracle in self.oracles.items():
                if oracle:
                    oracle.delete()
            self.oracles = {}
        self.oracle = None
        self.boomer = None
        if self.poracle:
            self.poracle.delete()
            self.poracle = None

    def predict(self, sample):
        '''
            make the prediction for the given sample via the prediction SAT oracle
        '''
        # translate the sample into literal assumption
        self.hypos = self.boomer.mxe.get_literals(sample)
        
        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, cat in enumerate(self.fcats):
            for v in range(cat[0], cat[1] + 1):
                self.v2cat[self.hypos[v]] = i
        
        # make the prediction
        assert self.poracle.solve(assumptions=self.hypos), 'Formula must be SAT!'
        model = self.poracle.get_model()
        
        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0
            for lit, wght in self.boomer.mxe.enc[clid].heads:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght
        # returning the class corresponding to the max score
        if self.verbose:
            print('log: SATExplainer, poracle score = {}'.format(list(scores.items()), key=lambda t: t[1]))
        return max(list(scores.items()), key=lambda t: t[1])[0]
    
    def prepare(self, sample):
        '''
            prepare the (K-1) SAT oracles to compute an explanation
        '''
        self.out_id = self.predict(sample)

        # Option 2 prepare the corresponding (K-1) SAT oracles via SATReasonerBR
        self.oracle = SATReasonerBR(self.formula, self.out_id, solver=self.optns.solver, 
            pb_encoding=self.optns.pb_encoding, wd=self.optns.w_digits, preprocess=self.optns.formula_preprocess)
        # self.oracles[self.out_id] = self.oracle

        # Option 1
        # self.oracle = self.oracles[self.out_id]

        # get the real encoding size
        self.size_formula = self.oracle.size_formula

        # transformed sample
        self.sample = list(self.boomer.transform_x_cases(sample)[0])

        # correct class id (corresponds to the maximum computed)
        self.output = self.boomer.target_name[self.out_id]

        if self.verbose:
            # preparing the output info for (sample, predicted_class) to explain!
            inpvals = self.boomer.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.boomer.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} == {1}'.format(f, v))
                else:
                    self.preamble.append(str(v))

            # print('| Explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample):
        '''
            hypotheses (explanation) minimization
        '''
        ######################################################################
        start_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        ######################################################################

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        ######################################################################
        time_prpare = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
        # print('log: prepared_time = {}'.format(time_prpare))
        ######################################################################

        # calling the actual explanation procedure
        self._explain(sample, xtype=self.optns.xtype, xnum=self.optns.xnum, reduce_=self.optns.reduce)

        ######################################################################
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time - time_prpare 

        self.used_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem
        ######################################################################

        if self.verbose:
            print('------------- SATExplainerBR ----------------')
            for expl in self.expls:
                hyps = list(reduce(lambda x, y: x + self.hypos[y[0]:y[1]+1], [self.fcats[c] for c in expl], []))
                expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                # print('expl = {}'.format(expl))
                preamble = [self.preamble[i] for i in expl]
                label = self.boomer.target_name[self.out_id]

                if self.optns.xtype in ('contrastive', 'con'):
                    preamble = [l.replace('==', '!=') for l in preamble]
                    label = 'NOT {0}'.format(label)

                # print('| Explained: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                print('  # hypos left:', len(expl))
            print('  calls:', self.calls)   # self.calls records the number of the calling the get_coex()
            print('  rtime: {0:.2f}'.format(self.time))
            print('  used_mem: {0:.2f}'.format(self.used_mem))

        return self.expls

    def _explain(self, sample, xtype='abd', xnum=1, reduce_='none'):
        '''
            The real extraction of explanations
            251203: currently, only support the set-minimal AXp extraction (not cardinality-smallest)
        '''
        if xtype in ('abductive', 'abd'):
            # Axp => MUS (Minimal UnSAT Subset) extraction
            if xnum == 1:
                # print('log: Computing 1 AXp!')
                self.expls = [self.extract_mus(reduce_=reduce_)]

    def extract_mus(self, reduce_='lin'):
        '''
            extract an Axp
        '''

        def _do_linear(core):
            '''
                the linear MUS extraction
            '''
            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    if not self.oracle.get_coex(self._cats2hypos(to_test)):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        # prepare for categorical case
        self.fcats = self.fcats_copy[:]

        # get the core obtained via SAT oracles
        assert self.oracle.get_coex(self.hypos) == None, 'No prediction'
        core = self.oracle.get_reason(self.v2cat)

        if self.verbose > 2:
            # print('core of size {}, is {}'.format(len(core), core))
            print('after cats2hypos len = {}'.format(len(self._cats2hypos(core))))
        self.calls = 1      # already 1 call to get the core => to minimize the candidate features to delete in explanation

        if reduce_ == 'none' or reduce_ == 'lin':
            # in default, making the linear MUS extraction
            expl = _do_linear(core)
        
        return expl

    def _cats2hypos(self, scats):
        """
            Translate selected categories into propositional hypotheses.
            note: scats => selected categories (target: to deal with the categorical encoding)
        """
        return list(reduce(lambda x, y: x + self.hypos[y[0] : y[1] + 1],
            [self.fcats[c] for c in scats], []))
    
#
#==============================================================================
class MXExplainerBR(object):
    """
        MaxSAT-based Explainer for Boosted Rules
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, boomer):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()
        self.fcats = []
        self.formula = formula

        # saving Boomerer
        self.boomer = boomer

        self.verbose = self.optns.verb

        # MaxSAT-based oracles
        self.oracles = {}

        # initialize the MaxSAT oracles 
        # As when the explainer initializes, no infos about the (sample, predict_class)
        # each oracles[clid] contains (clid - 1) MaxSAT oracles (The \sum_{i} - \sum_{clid})
        # for clid in range(nof_classes):
            # # print("log: MXReasonBR of class={}".format(clid))
            # self.oracles[clid] = MXReasonerBR(formula, clid,
            #         solver=self.optns.solver, oracle=ortype, 
            #         am1=self.optns.am1, exhaust=self.optns.exhaust,
            #         minz=self.optns.minz, trim=self.optns.trim)
            #         # verbose=self.optns.verb)

        # a reference to the current oracle
        self.oracle = None

        # SAT-based predictor
        # The usage of SAT oracle is to make the prediction
        self.poracle = SATSolver(name='g3')
        for clid in range(nof_classes):
            # LOG: possible duplication in clauses? (in 'common' before?)
            self.poracle.append_formula(formula[clid].formula)

        # determining which features should go hand in hand
        categories = collections.defaultdict(lambda: [])
        for f in self.boomer.extended_feature_names_as_array_strings:
            if f in self.ivars:
                if '_' in f or len(self.ivars[f]) == 2:
                    categories[f.split('_')[0]].append(self.boomer.mxe.vpos[self.ivars[f][0]])
                else:
                    for v in self.ivars[f]:
                        # this has to be checked and updated
                        categories[f].append(self.boomer.mxe.vpos[abs(v)])

        # these are the result indices of features going together
        self.fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
        self.fcats_copy = self.fcats[:]

        # all used feature categories
        self.allcats = list(range(len(self.fcats)))

        # variable to original feature index in the sample
        self.v2feat = {}
        for var in self.boomer.mxe.vid2fid:
            feat, ub = self.boomer.mxe.vid2fid[var]
            self.v2feat[var] = int(feat.split('_')[0][1:])

        # number of oracle calls involved
        self.calls = 0

    def __del__(self):
        """
            Destructor.
        """
        self.delete()

    def delete(self):
        """
            Actual destructor.
        """
        # deleting MaxSAT-based reasoners
        if self.oracles:
            for clid, oracle in self.oracles.items():
                if oracle:
                    oracle.delete()
            self.oracles = {}
        self.oracle = None

        # deleting the SAT-based predictor
        if self.poracle:
            self.poracle.delete()
            self.poracle = None

    def predict(self, sample):
        """
            Run the encoding and determine the corresponding class.
            => Decide by the model got via SAT call
        """

        # translating sample into assumption literals
        self.hypos = self.boomer.mxe.get_literals(sample)
        # print('log: hypos are {}'.format(self.hypos))

        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, cat in enumerate(self.fcats):
            for v in range(cat[0], cat[1] + 1):
                self.v2cat[self.hypos[v]] = i

        # running the solver to propagate the prediction;
        # using solve() instead of propagate() to be able to extract a model
        assert self.poracle.solve(assumptions=self.hypos), 'Formula must be satisfiable!'
        model = self.poracle.get_model()

        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in self.boomer.mxe.enc[clid].heads:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght
                
        # returning the class corresponding to the max scores
        return max(list(scores.items()), key=lambda t: t[1])[0]

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        # first, we need to determine the prediction, according to the model
        self.out_id = self.predict(sample)

        # 251029: instead of initializing all (K-1) MaxSAT oracles for different class (K*(K-1)),
        # we just create K-1 MaxSAT oracles for the target one
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'
        self.oracle = MXReasonerBR(self.formula, self.out_id,
                    solver=self.optns.solver, oracle=ortype, 
                    am1=self.optns.am1, exhaust=self.optns.exhaust,
                    minz=self.optns.minz, trim=self.optns.trim)
        self.oracles[self.out_id] = self.oracle
        # get the real encoding size
        self.size_formula = self.oracle.size_formula

        # transformed sample
        self.sample = list(self.boomer.transform_x_cases(sample)[0])

        # correct class id (corresponds to the maximum computed)
        self.output = self.boomer.target_name[self.out_id]

        if self.verbose:
            # preparing the output info for (sample, predicted_class) to explain!
            inpvals = self.boomer.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.boomer.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} == {1}'.format(f, v))
                else:
                    self.preamble.append(str(v))

            # print('| Explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """
        ######################################################################
        start_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime
        ######################################################################

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        ######################################################################
        time_prpare = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
        # print('log: prepared_time = {}'.format(time_prpare))
        ######################################################################

        # if self.optns.encode != 'mxe':
        #     # dummy call with the full instance to detect all the necessary cores
        #     self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True)

        # ######################################################################
        # time_dummy = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #         resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_prpare
        # print('log: dummy-call_time = {}'.format(time_dummy))
        # ######################################################################

        # calling the actual explanation procedure
        self._explain(sample, smallest=smallest, xtype=self.optns.xtype,
                xnum=self.optns.xnum, unit_mcs=self.optns.unit_mcs,
                reduce_=self.optns.reduce)

        ######################################################################
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time - time_prpare 

        self.used_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem
        ######################################################################

        if self.verbose:
            print('------------- MaxSATExplainerBR ----------------')
            for expl in self.expls:
                hyps = list(reduce(lambda x, y: x + self.hypos[y[0]:y[1]+1], [self.fcats[c] for c in expl], []))
                expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                # print('expl = {}'.format(expl))
                preamble = [self.preamble[i] for i in expl]
                label = self.boomer.target_name[self.out_id]

                if self.optns.xtype in ('contrastive', 'con'):
                    preamble = [l.replace('==', '!=') for l in preamble]
                    label = 'NOT {0}'.format(label)

                # print('| Explained: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                print('  # hypos left:', len(expl))

            print('  calls:', self.calls)   # self.calls records the number of the calling the get_coex()
            print('  rtime: {0:.2f}'.format(self.time))
            print('  used_mem: {0:.2f}'.format(self.used_mem))

        return self.expls

    def _explain(self, sample, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, reduce_='none'):
        """
            Compute an explanation.
        """
        # print('log: entering the true explaining!')
        if xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if not smallest and xnum == 1:
                # print('log: Computing 1 AXp!')
                self.expls = [self.extract_mus(reduce_=reduce_)]
            else:
                if self.verbose:
                    print('log: Computing the {} smallest AXp!'.format(xnum))
                self.mhs_mus_enumeration(xnum, smallest=smallest)
        else:  # contrastive explanations => MCS enumeration
            if self.verbose:
                print('log: Computing the {} CXps!'.format(xnum))
            self.mhs_mcs_enumeration(xnum, smallest, reduce_)

    def extract_mus(self, reduce_='lin', start_from=None):
        """
            Compute one abductive explanation.
        """

        def _do_linear(core):
            """
                Do linear search.
            """
            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    if not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_quickxplain(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    # actual binary hypotheses to test
                    self.calls += 1
                    if to_test and not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        self.fcats = self.fcats_copy[:]

        # this is our MUS over-approximation
        if start_from is None:
            assert self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True) == None, 'No prediction'

            # getting the core
            core = self.oracle.get_reason(self.v2cat)
        else:
            core = start_from

        if self.verbose > 2:
            # print('core of size {}, is {}'.format(len(core), core))
            print('after cats2hypos len = {}'.format(len(self._cats2hypos(core))))

        self.calls = 1  # we have already made one call

        if reduce_ == 'qxp':
            expl = _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            expl = _do_linear(core)

        return expl

    def mhs_mus_enumeration(self, xnum, smallest=False):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.allcats], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            if self.optns.unit_mcs:
                for c in self.allcats:
                    self.calls += 1
                    if self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                        hitman.hit([c])
                        self.duals.append([c])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                hypos = self._cats2hypos(hset)
                coex = self.oracle.get_coex(hypos, early_stop=True)
                if coex:
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hypos).difference(set(hypos)))

                    for h in removed:
                        if coex[abs(h) - 1] != h:
                            unsatisfied.append(self.v2cat[h])
                        else:
                            hset.append(self.v2cat[h])

                    unsatisfied = list(set(unsatisfied))
                    hset = list(set(hset))

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.get_coex(self._cats2hypos(hset + [h]), early_stop=True):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.duals.append([to_hit])
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def mhs_mcs_enumeration(self, xnum, smallest=False, reduce_='none', unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.allcats], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for c in self.allcats:
                self.calls += 1

                if not self.oracle.get_coex(self._cats2hypos([c]), early_stop=True):
                    hitman.hit([c])
                    self.duals.append([c])
                elif unit_mcs and self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([c])
                    self.expls.append([c])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.get_coex(self._cats2hypos(set(self.allcats).difference(set(hset))), early_stop=True):
                    to_hit = self.oracle.get_reason(self.v2cat)

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def _cats2hypos(self, scats):
        """
            Translate selected categories into propositional hypotheses.
            note: scats => selected categories (target: to deal with the categorical encoding)
        """
        return list(reduce(lambda x, y: x + self.hypos[y[0] : y[1] + 1],
            [self.fcats[c] for c in scats], []))

    def _hypos2cats(self, hypos):
        """
            Translate propositional hypotheses into a list of categories.
        """

        pass
