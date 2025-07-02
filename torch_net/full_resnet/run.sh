#!/bin/bash

./cleanup.sh
sbatch train.sh
tail -f -n 100 -s 0.25 dp_resnet.err
