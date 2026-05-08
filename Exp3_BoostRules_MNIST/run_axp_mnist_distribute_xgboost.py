#!/usr/bin/env python
##
## experiment => lauching selected depth + num
##
## Distributed version of computing AXps from MNIST instance, when a single one needs large memory.
## Step 1: generate the command of running all AXps for different MNIST instance, save the results into separate outfile
## Step 2: run the commands distribute
## Step 3: grep all information in output file together
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
from xgbooster import XGBooster
from options import Options

#
#==============================================================================
def parse_options():
    """
        Standard options handling.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:hi:n:r:v:p:e:w:m:t:',
                ['depth=', 'help', 'inst=',  'num=', 'relax=', 'verbose', 'pbencoding=', 'encoding=', 'wd=', 'mode=', 'tl='])
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

    # 260226: add the mode option for generating commands (mode = 0), and wrapper information (mode = 1)
    mode = None
    time_limit = 7200.0

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
        elif opt in ('-m', '--mode'):
            mode = int(arg)
            assert mode in [0, 1, 2]
        elif opt in ('-t', '--tl'):
            time_limit = float(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return depth, num, inst, relax, verbose, encoding, pb_encoding, wd, mode, time_limit, args

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
    avg_l = lambda l: sum(l) / len(l)

    depth, num, count, relax, verbose, encoding, pb_encoding, wd, mode, tl, file = parse_options() 
    if file:
        # file can be reused in inst_selected.csv
        assert os.path.exists(file[0])
        data_path = file[0]
        count = int(count)
    else:
        # just pass the mnist_train
        data_path = '../bench_mnist_csv/mnist_train.csv'

    if (encoding is None) and (wd is None) and (pb_encoding is None):
        results_folder = 'results_AXp_xgboost_orig'+ '/mx_md' + str(depth) + '_num' + str(num)
    else: 
        results_folder = 'results_AXp_xgboost_' + encoding + '/mx_md' + str(depth) + '_num' + str(num) + '_wd' + str(wd) + '_pb' + str(pb_encoding)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # reading the selected instances
    with open(os.path.join(data_path), 'r') as fp:
        insts = [line.strip().rsplit(',', 1)[0] for line in fp.readlines()[1:]]
    base = os.path.splitext(os.path.basename(data_path))[0]

    # preparing the model file path
    if (encoding is None) and (wd is None) and (pb_encoding is None):
        mfile = './mnist_xgboost_260116/{0}/{0}_nbestim_{1}_maxdepth_{2}_testsplit_0.2.mod.pkl'.format(base, num, depth)
    else:
        # 251209: thinking about different wd
        mfile = './mnist_xgboost_260304/{0}/{0}_nbestim_{1}_maxdepth_{2}_testsplit_0.2_wd_{3}.mod.pkl'.format(base, num, depth, wd)

    log_folders = os.path.join(os.path.join(results_folder, 'logs'))
    if not os.path.exists(log_folders):
        os.makedirs(log_folders)

    # preparing the commands
    if (encoding is None) and (wd is None) and (pb_encoding is None):
        command_options = f'xreason.py --relax ' + str(relax) + ' -s g3 -z -X abd -R lin -u -N 1 -e mx --fpreprocess -x \'inst\' somefile' 
        command = 'python3 run_axp_mnist_distribute_xgboost.py -m 1 -n ' + str(num) + ' -d ' + str(depth)
    else:
        # 251209: apply the new used options with wd, pb_encoding, encoding
        command_options = f'xreason.py --relax ' + str(relax) + ' -s g3 -z -X abd -R lin -u -N 1 -e ' + encoding + ' --wdigit ' + str(wd) +\
                        ' --pbencoding ' + str(pb_encoding) + ' --fpreprocess -x \'inst\' somefile'
        command = 'python3 run_axp_mnist_distribute_xgboost.py -m 1 -n ' + str(num) + ' -d ' + str(depth) + ' -e sat -p ' \
            + str(pb_encoding) + ' -w ' + str(wd)

    # prepare the options + mxboomer
    moptions = Options(command_options.split())
    mxgb = XGBooster(moptions, from_model=mfile)

    if mode == 0:
        # mode == 0 => generate the commands list saved in file + generate the instance list in file
        # shuffling the instances  
        random.Random(1234).shuffle(insts)
        if count > 1:
            nof_insts = min(int(count), len(insts))
        else:
            nof_insts = min(int(len(insts) * count), len(insts))
        print(f'considering {nof_insts} instances')

        instance_unique = []
        for i, inst in enumerate(insts):
            if len(instance_unique) == nof_insts:
                break
            if not inst in instance_unique:
                # # 251209: #BUG found
                # # before explaining the instance, we need judge that the scores of the instance, it should be the clear biggest one.
                # # where 2 or more classes share the same biggest score is not allowed 
                # # it would be failed to explain the prediction, therefore just ignore this instance
                # inst_e = np.array([float(v.strip()) for v in inst.split(',')])
                # s_pred_scores = sorted(mxboomer.predict_scores(inst_e), reverse=True)
                # if math.isclose(s_pred_scores[0], s_pred_scores[1], abs_tol=1e-12):
                #     continue
                instance_unique.append(inst)
            else:
                continue
        # write the instance_unique into the file base_inst.csv
        with open(os.path.join(log_folders, base + '.csv'), 'w') as f:
            f.write('nothing! just avoid the title line!\n')
            for inst in instance_unique:
                f.write(str(inst) + '\n')

        # write the commands into the base_commands.txt
        with open(os.path.join(log_folders, base + '_commands.txt'), 'w') as f:
            for i in range(len(instance_unique)):
                f.write(command + ' -i ' + str(i) + ' ' + os.path.join(log_folders, base+'.csv') + '\n')
        # using xargs command to execute all commands in the commands.txt
    elif mode == 1:
        print('log: mode == 1, explaining instance ' + str(count))
        # running the Axp computation for a single instance => attention, here count indicates the instance selected to explain
        mxgb.encode(test_on=None)
        print('log: mxgb, encode finished!')

        moptions.explain = [float(v.strip()) for v in insts[count].split(',')]
        expl2 = mxgb.explain(moptions.explain)
        print('log: mxgb, explanation finished!')

        size_formulas = mxgb.x.size_formula
        with open(os.path.join(log_folders, base + '_i' + str(count) + '.log'), 'w') as f:
            f.write(f'avg vars: {avg_l([i[0] for i in size_formulas.values()]):.2f}' + '\n')
            f.write(f'avg cos: {avg_l([i[1] for i in size_formulas.values()]):.2f}' + '\n')
            f.write(f'avg cps : {avg_l([i[2] for i in size_formulas.values()]):.2f}' + '\n')
            f.write(f'i: {insts[count]}' + '\n')
            f.write(f's: {len(expl2[0])}' + '\n')
            f.write(f't: {mxgb.x.time:.3f}' + '\n')
            f.write(f'c: {mxgb.x.calls}' + '\n')
            f.write(f'axp: {expl2}' + '\n')
    else:
        # generating the summary log in the result_folder/base.log
        avg_vars_all, avg_cos_all, avg_cps_all, time_all, axplens = [], [], [], [], []

        logs = [f for f in os.listdir(log_folders) if f.endswith('.log')]

        # check whether all commands are finished
        unfinshed_commands = []
        with open(os.path.join(log_folders, base + '_commands.txt'), 'r') as f:
            for c in f.readlines():
                i_str = base + '_i' + c.split('-i ')[1].split(' ')[0] + '.log'
                if not i_str in logs:
                    unfinshed_commands.append(i_str)

        if len(unfinshed_commands) > 0:
            print('Commands not finished, including: ' + str(unfinshed_commands))
        # else:
        #     print('Commands all finsihed!')

        with open(os.path.join(results_folder, base + '.log'), 'w') as fbase: 
            for l in logs:
                with open(os.path.join(log_folders, l), 'r') as fl:
                    for info in fl.readlines():
                        if info.startswith('t'):
                            t = float(info.split('t: ')[-1])
                            if t < tl:
                                time_all.append(t)
                                fbase.write(info)
                        elif info.startswith('avg vars'):
                            avg_vars_all.append(float(info.split('avg vars: ')[-1]))
                        elif info.startswith('avg cos'):
                            avg_cos_all.append(float(info.split('avg cos: ')[-1]))
                        elif info.startswith('avg cps'):
                            avg_cps_all.append(float(info.split('avg cps : ')[-1]))
                        elif info.startswith('s'):
                            axplens.append(int(info.split('s: ')[-1]))
                        else:
                            fbase.write(info)
                fbase.write('\n')
            # write the min/max/avg in time and formula_size
            fbase.write(f'avg vars: {avg_l(avg_vars_all):.2f}\n')
            fbase.write(f'avg cos: {avg_l(avg_cos_all):.2f}\n')
            fbase.write(f'avg cps : {avg_l(avg_cps_all):.2f}\n\n')

            fbase.write(f'num time: {len(time_all)}\n')
            fbase.write(f'max time: {max(time_all):.2f}\n')
            fbase.write(f'min time: {min(time_all):.2f}\n')
            fbase.write(f'avg time: {avg_l(time_all):.2f}\n')
            fbase.write(f'avg axplen: {avg_l(axplens):.2f}\n')


                


