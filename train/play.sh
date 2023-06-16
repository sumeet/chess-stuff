#!/bin/bash
dir="$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
cd $dir
python ./play.py 2>>output.play

