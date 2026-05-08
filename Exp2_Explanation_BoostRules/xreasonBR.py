#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xreasonBR.py
##
##  Created on: Aug 26, 2025
#
#==============================================================================
from __future__ import print_function

import os
import sys

sys.path.append("..")
from boomerer.boomerer import Boomerer
from boomerer.preprocess import preprocess_dataset
from data import Data
# from anchor_wrap import anchor_call
# from lime_wrap import lime_call
# from shap_wrap import shap_call
from options import Options

#
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    if (options.preprocess_categorical):
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    if options.files:
        boomer = None

        if options.train:
            print("log: no training!!")
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)

            boomer = Boomerer(options, from_data=data)
            train_accuracy, test_accuracy, model = boomer.train()

        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]
            
        if options.encode:
            if not boomer:
                boomer = Boomerer(options, from_model=options.files[0])
            
            # encode it and save the encoding to another file
            boomer.encode(test_on=options.explain)

        if options.explain:
            if not boomer:
                if options.uselime or options.useanchor or options.useshap:
                    boomer = Boomerer(options, from_model=options.files[0])
                else:
                    # abduction-based approach requires an encoding
                    boomer = Boomerer(options, from_encoding=options.files[0])

            # checking LIME or SHAP should use all features
            if not options.limefeats:
                options.limefeats = len(data.names) - 1

            # explain using anchor or the abduction-based approach
            expl = boomer.explain(options.explain)
                    # use_lime=lime_call if options.uselime else None,
                    # use_anchor=anchor_call if options.useanchor else None,
                    # use_shap=shap_call if options.useshap else None,
                    # nof_feats = options.limefeats)

            # if (options.uselime or options.useanchor or options.useshap) and options.validate:
            #     boomer.validate(options.explain, expl)
