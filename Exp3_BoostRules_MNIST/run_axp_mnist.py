#!/usr/bin/env python
##
## experiment => lauching selected depth + num
##
##  Created on: Aug 27, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##  Changed on 251006
##      Author: Hao Hu
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
import random
import shutil
import subprocess
import sys
import resource
import time
import numpy as np

sys.path.append("..")
from boomerer.boomerer import Boomerer
from options import Options

#
#==============================================================================
def parse_options():
    """
        Standard options handling.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:hi:n:r:v:p:e:w:',
                ['depth=', 'help', 'inst=',  'num=', 'relax=', 'verbose', 'pbencoding=', 'encoding=', 'wd='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    depth = 5
    inst = 0.3
    num = 50
    relax = 0
    verbose = False

    # 251209: add the options with weight_digit and pb_encoding, approach, 
    encoding = None # sat => SAT appraoch, mx => MaxSAT approach
    pb_encoding = None  # 0 => 'Best', 1 => 'BDD', etc (check options.py)
    wd = None           # decimal places used for weight digits

    for opt, arg in opts:
        if opt in ('-d', '--depth'):
            depth = str(arg)
            if depth == 'none':
                depth = -1
            else:
                depth = int(depth)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-i', '--inst'):
            inst = float(arg)
        elif opt in ('-n', '--num'):
            num = int(arg)
        elif opt in ('-r', '--relax'):
            relax = int(arg)
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt in ('-p', '--pbencoding'):
            pb_encoding = int(arg)
        elif opt in ('-e', '--encoding'):
            encoding = str(arg)
        elif opt in ('-w', '--wd'):
            wd = int(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return depth, num, inst, relax, verbose, encoding, pb_encoding, wd, args

#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] datasets-list')
    print('Options:')
    print('        -d, --depth=<int>         Tree depth')
    print('                                  Available values: [1 .. INT_MAX], none (default = 5)')
    print('        -h, --help                Show this message')
    print('        -i, --inst=<float,int>    Fraction or number of instances to explain')
    print('                                  Available values: (0 .. 1] or [1 .. INT_MAX] (default = 0.3)')
    print('        -n, --num=<int>           Number of trees per class')
    print('                                  Available values: [1 .. INT_MAX] (default = 50)')
    print('        -r, --relax=<int>         Relax decimal points precision to this number')
    print('                                  Available values: [0 .. INT_MAX] (default = 0)')
    print('        -v, --verbose             Be verbose')
    print('        -e, --encoding            Encoding chosen: mx => MaxSAT, sat => SAT')
    print('        -p, --pbencoding         PB constraint encoding type')
    print('        -w, --wd                 The decimal place used to approximate the weight in Boosted rules')

#
#==============================================================================
if __name__ == '__main__':
    depth, num, count, relax, verbose, encoding, pb_encoding, wd, files = parse_options()

    if files:
        datasets = files[0]
    else:
        # 251014: take care, just for spambase, change later => done
        datasets = 'datasets.list'

    with open(datasets, 'r') as fp:
        datasets = [line.strip() for line in fp.readlines() if line]

    print(f'training parameters: {num} trees per class, each of depth {"adaptive" if depth == -1 else depth}\n')

    # deleting the previous results
    # if os.path.isdir('results'):
    #     shutil.rmtree('results')
    if (encoding is None) and (wd is None) and (pb_encoding is None):
        results_folder = 'results_AXp_boomer_orig'+ '/mx_md' + str(depth) + '_num' + str(num)
    else: 
        results_folder = 'results_AXp_boomer_' + encoding + '/mx_md' + str(depth) + '_num' + str(num) + '_wd' + str(wd) + '_pb' + str(pb_encoding)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # initializing the seed
    # random.seed(1234)
    if (encoding is None) and (wd is None) and (pb_encoding is None):
        command_options = f'../Exp2_Explanation_BoostRules/xreasonBR.py --relax ' + str(relax) + ' -s g3 -z -X abd -R lin -u -N 1 -e mx --drule --fpreprocess -x \'inst\' somefile'
    else:
        # 251209: apply the new used options with wd, pb_encoding, encoding
        command_options = f'../Exp2_Explanation_BoostRules/xreasonBR.py --relax ' + str(relax) + ' -s g3 -z -X abd -R lin -u -N 1 -e ' + encoding + ' --wdigit ' + str(wd) +\
                        ' --pbencoding ' + str(pb_encoding) + ' --drule --fpreprocess -x \'inst\' somefile'
    moptions = Options(command_options.split())

    # training all XGBoost models
    for data in datasets:
        if depth != -1:
            adepth = depth
            data = data.split()[0]
        else:
            # adaptive length
            data, adepth = data.split()

        print(f'processing {data}...')

        # reading and shuffling the instances
        with open(os.path.join(data), 'r') as fp:
            insts = [line.strip().rsplit(',', 1)[0] for line in fp.readlines()[1:]]
            # insts = sorted(list(set(insts)))
            random.Random(1234).shuffle(insts)

            if count > 1:
                nof_insts = min(int(count), len(insts))
            else:
                nof_insts = min(int(len(insts) * count), len(insts))
            print(f'considering {nof_insts} instances')

        base = os.path.splitext(os.path.basename(data))[0]
        if (encoding is None) and (wd is None) and (pb_encoding is None):
            mfile = './mnist_boomers_251119/{0}/{0}_maxrule_{1}_maxcond_{2}_testsplit_0.2.mod.pkl'.format(base, num, adepth)
        else:
            # 251209: thinking about different wd
            mfile = './mnist_boomers_251119/{0}/{0}_maxrule_{1}_maxcond_{2}_testsplit_0.2_wd_{3}.mod.pkl'.format(base, num, adepth, wd)

        mlog = open(os.path.join(results_folder, base + '.log'), 'w')

        # creating booster objects
        mxboomer = Boomerer(moptions, from_model=mfile)
        mxboomer.encode(test_on=None)
        print('log: encode finished!!')

        mtimes, mcalls, mxmem, axplens = [], [], [], []
        instance_unique = []
        size_formulas = None
   
        for i, inst in enumerate(insts):
            # 251013: change to avoid possible duplicated instances
            if len(instance_unique) == nof_insts:
                print("log: 100% instances explained!")
                break
            
            if i == len(insts) - 1:
                # treat the case showed before
                print("log: 100% instances explained!")

            if len(instance_unique) == nof_insts / 2:
                print("log: 50% instances explained!")
            
            if not inst in instance_unique:
                # 251209: #BUG found
                # before explaining the instance, we need judge that the scores of the instance, it should be the clear biggest one.
                # where 2 or more classes share the same biggest score is not allowed 
                # it would be failed to explain the prediction, therefore just ignore this instance
                inst_e = np.array([float(v.strip()) for v in inst.split(',')])
                s_pred_scores = sorted(mxboomer.predict_scores(inst_e), reverse=True)
                if math.isclose(s_pred_scores[0], s_pred_scores[1], abs_tol=1e-12):
                    continue
                instance_unique.append(inst)
            else:
                continue

            # processing the instance
            moptions.explain = [float(v.strip()) for v in inst.split(',')]
            expl2 = mxboomer.explain(moptions.explain)
            print('log: {} is explained!!', i)

            # 251209: update the way of getting encoding information
            if size_formulas is None:
                size_formulas = mxboomer.x.size_formula # it should contains the information of each oracle
                avg_l = lambda l: sum(l) / len(l)
                print(f'avg vars: {avg_l([i[0] for i in size_formulas.values()]):.2f}', file=mlog)
                print(f'avg cos: {avg_l([i[1] for i in size_formulas.values()]):.2f}', file=mlog)
                print(f'avg cps : {avg_l([i[2] for i in size_formulas.values()]):.2f}', file=mlog)

            print(f'i: {inst}', file=mlog)
            print(f's: {len(expl2[0])}', file=mlog)
            print(f't: {mxboomer.x.time:.3f}', file=mlog)
            print(f'c: {mxboomer.x.calls}', file=mlog)

            #
            mxmem.append(round(mxboomer.x.used_mem / 1024.0, 3))
            mtimes.append(mxboomer.x.time)
            mcalls.append(mxboomer.x.calls)
            axplens.append(expl2[0])
            #
            mxboomer.x.calls = 0
            
            # 251210: a try to reduce the memory used
            # mxboomer.x.delete_oracle()
            
            # print(f"mem usage: SMT={smem[-1]} MB MaxSAT={mxmem[-1]} MB")
            print(f'mem: {mxmem[-1]}', file=mlog)
            
            # 251006: change for adding the mem info in logs
            print('', file=mlog)

            mlog.flush()
            sys.stdout.flush()

        ##################

        print(f"max time: {max(mtimes):.2f}", file=mlog)
        print(f"min time: {min(mtimes):.2f}", file=mlog)
        print(f"avg time: {sum(mtimes)/len(mtimes):.2f}", file=mlog)
        print('', file=mlog)
        print(f"avg calls: {sum(mcalls)/len(mcalls):.2f}", file=mlog)
        print(f"avg axplens: {sum(axplens)/len(axplens):.2f}", file=mlog)
        #################
        mlog.close()

        print('done')
