import argparse
import glob
import logging
import os

import fs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tenacity import retry, wait_exponential

from utils import remove_extension

def setup_logging():
    logging.basicConfig(
        filename="audio_processing.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

@retry(wait=wait_exponential(multiplier=1, min=4, max=120))
def do_connection(connection_string):
    """Establish a connection to the filesystem with retries."""
    try:
        if connection_string:
            return fs.open_fs(connection_string)
        return False
    except Exception as e:
        logging.error(f"Attempt failed to connect to filesystem: {e}")
        logging.info("Retrying connection...")
        raise

def walk_audio(filesystem, input_path):
    """Walk through the filesystem and yield audio files."""
    walker = filesystem.walk(
        input_path,
        filter=["*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a", "*.WAV", "*.MP3"],
    )
    for path, _dirs, flist in walker:
        for f in flist:
            yield fs.path.combine(path, f.name)

def parse_folders(filesystem, apath, rpath):
    """
    Parse audio and result folders, matching audio files with their corresponding result files.
    """
    audio_files = get_audio_files(filesystem, apath)
    audio_no_extension = [remove_extension(audio_file) for audio_file in audio_files]

    result_files = [
        f
        for f in glob.glob(rpath + "/**/*", recursive=True)
        if os.path.isfile(f)
    ]
    matched_files = match_audio_and_results(
        audio_files, audio_no_extension, result_files
    )

    logging.info(f"Found {len(matched_files)} audio files with valid result file.")
    return matched_files

def get_audio_files(filesystem, apath):
    """Get all audio files from the specified path."""
    if not filesystem:
        audio_files = [
            f
            for f in glob.glob(apath + "/**/*", recursive=True)
            if os.path.isfile(f)
        ]
        return [f for f in audio_files if f.endswith((".WAV", ".wav", ".mp3"))]
    else:
        return [audiofile for audiofile in walk_audio(filesystem, apath)]

def match_audio_and_results(audio_files, audio_no_extension, result_files):
    """Match audio files with their corresponding result files."""
    matched_files = []
    for result in result_files:
        result_no_extension = remove_extension(result)
        if result_no_extension in audio_no_extension:
            audio_idx = audio_no_extension.index(result_no_extension)
            matched_files.append({"audio": audio_files[audio_idx], "result": result})
    return matched_files

def parse_files(file_list, max_segments=10, threshold=0.6):
    """Parse the file list and make a list of segments."""
    segments = []
    for files in file_list:
        segments.extend(find_segments(files["audio"], files["result"], threshold))
    logging.info(f"Found {len(segments)} segments in total.")
    return segments

def find_segments(audio_file, result_file, confidence_threshold):
    """Find segments in the result file that meet the confidence threshold."""
    segments = []
    try:
        with open(result_file) as rf:
            lines = [line.strip() for line in rf.readlines()]

        for i, line in enumerate(lines):
            if i > 0:  # Skip header
                data = line.split("\t")
                start, end, species, confidence = (
                    float(data[3]),
                    float(data[4]),
                    data[7],
                    float(data[-3]),
                )
                if confidence >= confidence_threshold:
                    segments.append(
                        {
                            "audio": audio_file,
                            "start": start,
                            "end": end,
                            "species": species,
                            "confidence": confidence,
                        }
                    )

    except Exception as e:
        logging.error(f"Error processing result file {result_file}: {e}")
        raise e

    return segments

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_connection.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    myfs = do_connection(config["CONNECTION_STRING"])
    parsed_folders = parse_folders(myfs, config["INPUT_PATH"], config["OUTPUT_PATH_BIRDNET"])
    parsed_segments = parse_files(
        parsed_folders, max_segments=config["NUM_SEGMENTS"], threshold=config["THRESHOLD"]
    )

    # Create a Parquet table
    table = pa.table(
        {
            "audio": [segment["audio"] for segment in parsed_segments],
            "start": [segment["start"] for segment in parsed_segments],
            "end": [segment["end"] for segment in parsed_segments],
            "species": [segment["species"] for segment in parsed_segments],
            "confidence": [segment["confidence"] for segment in parsed_segments],
        }
    )

    # Write the table to a Parquet file
    pq.write_table(table, "sample.parquet")
    logging.info("Parquet file written successfully with all segments!")