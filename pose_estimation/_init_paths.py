# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        print(f"Adding {path} to sys.path")
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
#lib_path = osp.join(this_dir, '..', 'lib')
print("this_dir: ", this_dir)
parent_dir = osp.dirname(this_dir)
lib_path = osp.join(parent_dir, 'lib')
print("lib_path: ", lib_path)

add_path(lib_path)
