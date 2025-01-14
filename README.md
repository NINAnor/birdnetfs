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
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

2- Clone the BirdNET repository:
  
`birdnetfs` attempts to reuse `BirdNET` functions as much as possible.

```bash
git clone https://github.com/kahst/BirdNET-Analyzer.git
mv BirdNET-Analyzer birdnetsrc
```

3- Analyze

First, you need to change the **BirdNET config_file** (**NOT** the config_connection.yaml) located in `src/config.py`. More particularly you may want to change the `OUTPUT_PATH` parameter which is the path where the output files will be created.

There is two options.

- First you can analyze a file of your chosing using:

```bash
export PYTHONPATH="${PYTHONPATH}:./src/birdnetsrc:src:birdnetsrc"
python analyse.py filecache::ssh://$USER:$PASSWORD@HOST:/PATH/TO/AUDIO/FILE1.mp3
```

- Or analyze multiple files in parallel (using [GNU parallel](https://www.gnu.org/software/parallel/)):

In `files_to_analyze.csv` list the files that you want to analyze

Then run:

```bash
sudo apt-get install parallel
time systemd-run --scope --user --property=CPUWeight=1 -- sh -c './analyse.sh'
```

Analyzing the files will return `Birdnet.selection.table.txt` files in the `OUTPUT_PATH_BIRDNET`.

## Extract the detections

1- Update the `config_connection.yaml`

2- Build a `sample.parquet` database using:

```bash
python3 src/parse_results.py
```

This `parquet` database is a database containing ALL the results from BirdNET.


3- Build a `sampled_segments.parquet`

```bash
python3 src/global_sampler.py
```


:star: Note that this `parquet` file will contain `$NUM_SEGMENT` random segments with the `$THRESHOLD` indicated in the `config_connection.yaml`. 

4- Extract the detections!

```bash
./extract.sh
```

## Format for annotation

To annotate the extracted segments, we create a `csv` file per species. The `csv` list the segments to annotate and add a few columns for the annotator to fill. To help create the `csv` files, the repository contains a bash script `to_annotation_sheet.sh`.

Be sure to change the input and output in the script L4 and L5:

```bash
BASE_DIR= ./extracted_segments # Folder where all the segments are stored 
OUTPUT_DIR= ./csv # Folder that receives the csv files
```
Then run the script:

```bash
./to_annotation_sheet.sh
```




