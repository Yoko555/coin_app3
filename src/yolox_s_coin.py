#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # detect classes number of model
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "C:/Users/mi_yo/OneDrive/デスクトップ/YOLOX/datasets/Coin"
        # name of annotation file for training
        self.train_ann = "train.json"
        # name of annotation file for evaluation
        self.val_ann = "val.json"

