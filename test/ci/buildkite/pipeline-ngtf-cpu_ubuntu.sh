#!/bin/bash

# Taken from https://gist.github.com/toolmantim/1337952c8b5b1e5b0b5ea10c40e9efe4
#
# Outputs a pipeline that targets agents that have the same 'name' meta-data
# value as the step that does the pipeline upload. This means that all the
# steps will run on the same agent machine, assuming that the 'name' meta-data
# value is unique to each agent.
#
# Each agent needs to be configured with meta-data like so:
#
# meta-data="name=<unique-name>"
#
# To use, save this file as .buildkite/pipeline.sh, chmod +x, and then set your
# first pipeline step to run this and pipe it into pipeline upload:
#
# .buildkite/pipeline.sh | buildkite-agent pipeline upload
#

name=$(buildkite-agent meta-data get name)

cat << EOF
steps:
  - command: |
      rm -rf /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID
      virtualenv -p /usr/bin/python3 /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv 
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      pip install -U yapf==0.26.0
      
    label: ":gear: Setup"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait

  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      export PATH=/opt/llvm-3.9.0/bin/:$$PATH 
      maint/check-code-format.sh
      
    label: ":pencil: Code Format ?"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait

  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      python3 build_ngtf.py --artifacts /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID \
      --use_tensorflow_from_location /localdisk/buildkite-agent/prebuilt_tensorflow_1_15_2
      
    label: ":hammer_and_wrench: Build"
    timeout_in_minutes: 60
    agents:
      name: "$name"

  - wait

  - command: |
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID --test_cpp
    
    label: ":chrome: C++ Unit Test"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait 

  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      pip install psutil && pip install -U \
        /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/tensorflow/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
      pip install -U pip==19.3.1
      pip install -U /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/ngraph_tensorflow_bridge-*.whl
      
    label: ":gear: Install"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait
  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      pip install pytest
      pip install keras==2.3.1
      PYTHONPATH=`pwd`:`pwd`/tools:`pwd`/examples:`pwd`/examples/mnist python3 test/ci/buildkite/test_runner.py \
        --artifacts /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID --test_python
    
    label: ":python: nGraph Pytest"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait
  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      pip install pytest
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID --test_tf_python
    
    label: ":python: TensorFlow Pytest"
    timeout_in_minutes: 60
    agents:
      name: "$name"

  - wait

  - command: |
      source /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/venv/bin/activate 
      PYTHONPATH=`pwd` python3 test/ci/buildkite/test_runner.py \
        --artifacts /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID --test_resnet
    label: ":bar_chart: ResNet50"
    timeout_in_minutes: 30
    agents:
      name: "$name"

  - wait: ~

  - command: |
      rm -rf /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID
    label: ":wastebasket: Cleanup"
    agents:
      name: "$name"

EOF
