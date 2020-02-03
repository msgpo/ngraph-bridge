# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""nGraph TensorFlow bridge dynamic features test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
import numpy as np
from common import NgraphTest


class TestDynamic(NgraphTest):

    def test_api(self):
        assert not ngraph_bridge.is_dynamic(), "By default expected to be false"
        ngraph_bridge.use_dynamic()
        assert ngraph_bridge.is_dynamic()
        ngraph_bridge.use_static()
        assert not ngraph_bridge.is_dynamic()
