import argparse
import glob
import logging
import os
import traceback

import fs
import numpy as np
import pyarrow.parquet as pq
import yaml
from tenacity import retry, wait_exponential
from utils import openAudioFile, openCachedFile, saveSignal


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
        #logging.error(f"Attempt failed to connect to filesystem: {e}")
        #logging.info("Retrying connection...")
        raise

# @retry(wait=wait_exponential(multiplier=5, min=60, max=600))
def extract_segments(item, sample_rate, out_path, filesystem, seg_length=3):
    """Extract segments from the audio file and save them."""
    segments = item
    audio_file = item["audio"]

    signal, rate = (
        openAudioFile(audio_file, sample_rate)
        if not filesystem
        else openCachedFile(filesystem, audio_file, sample_rate)
    )

    save_extracted_segments(signal, rate, segments, out_path, seg_length)
    #logging.info(f"Segments extracted from {audio_file}")


def save_extracted_segments(signal, rate, segment, out_path, seg_length):
    """Save the extracted segments to the output path."""
    # for segment in segments:
    start = int(segment["start"] * rate)
    end = int(segment["end"] * rate)
    offset = ((seg_length * rate) - (end - start)) // 2
    start, end = max(0, start - offset), min(len(signal), end + offset)

    if end > start:
        segment_signal = signal[start:end]
        save_segment(segment_signal, segment, out_path)



def save_segment(segment_signal, segment, out_path):
    """Save an individual segment."""
    species_path = os.path.join(out_path, segment["species"])
    os.makedirs(species_path, exist_ok=True)

    segment_name = f"start={segment['start']}_end={segment['end']}_conf={segment['confidence']:.3f}_file={os.path.basename(segment['audio']).rsplit('.', 1)[0]}.wav"
    segment_path = os.path.join(species_path, segment_name)
    print(f"Segment {segment_path} saved")
    saveSignal(segment_signal, segment_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_connection.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--parquet_file", default="sample.parquet")
    parser.add_argument("audio_file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    myfs = do_connection(config["CONNECTION_STRING"])

    items = pq.read_table(args.parquet_file, filters=[["audio", "=", args.audio_file]])

    for item in items.to_pylist():
        print(f"Extracting segments from {item}")
        extract_segments(
            item, config["SAMPLE_RATE"], config["OUT_PATH_SEGMENTS"], myfs, seg_length=3
        )
