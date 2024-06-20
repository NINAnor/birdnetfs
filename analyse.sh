#!/bin/bash

export LC_ALL=C

# Change PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:./birdnetsrc"

# Files to process
FILE_LIST="files_to_analyze.csv"

# Using GNU Parallel to process each file
parallel --progress --eta --resume --joblog status.txt python analyse.py :::: $FILE_LIST
