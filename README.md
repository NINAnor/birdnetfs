# BIRDNETFS: BirdNET compatible with fsspec

## Introduction

[BirdNET](https://github.com/kahst/BirdNET-Analyzer) is great for analyzing files stored locally but we sometimes want to analyze files that are stored remotely. This repository aims to close that gap by using a modified version of [BirdNET-Analyzer/analyze.py](https://github.com/kahst/BirdNET-Analyzer/blob/main/analyze.py) that leverages [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), a python library that help open files that are stored remotely.

## How to use it?

This repository is made so that we can pull the changes made to the [BirdNET](https://github.com/kahst/BirdNET-Analyzer) repository without any conflict.

1- Clone this repository and install the dependancies

```bash
# Clone this repository:
git clone https://github.com/NINAnor/birdnetfs.git
cd birdnetfs
pip install requirements.txt
```

2- Clone the BirdNET repository:

```bash
git clone https://github.com/kahst/BirdNET-Analyzer.git
mv BirdNET-Analyzer birdnetsrc
```

3- Analyze

There is two options. 

- First you can analyze a file of your chosing using:

```bash
python analyse.py filecache::ssh://USER:PASSWORD@HOST:/PATH/TO/AUDIO/FILE1.mp3
```

- Or analyze multiple files in parallel (using [GNU parallel](https://www.gnu.org/software/parallel/)):

In `files_to_analyze.csv` list the files that you want to analyze

Then run:

```bash
sudo apt-get install parallel
time systemd-run --scope --user --property=CPUWeight=1 -- sh -c './analyse.sh'
```