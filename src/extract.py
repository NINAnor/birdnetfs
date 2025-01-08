import argparse
import logging
import os
import pandas as pd

import fs
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
    except Exception:
        # logging.error(f"Attempt failed to connect to filesystem: {e}")
        # logging.info("Retrying connection...")
        raise

# @retry(wait=wait_exponential(multiplier=5, min=60, max=600))
def extract_segments(
    item, sample_rate, out_path, filesystem, connection_string, seg_length=3
):
    """Extract segments from the audio file and save them."""
    segments = item
    audio_file = os.path.join(connection_string, item["audio"])

    signal, rate = (
        openAudioFile(audio_file, sample_rate)
        if not filesystem
        else openCachedFile(filesystem, audio_file, sample_rate)
    )

    save_extracted_segments(signal, rate, segments, out_path, seg_length)
    # logging.info(f"Segments extracted from {audio_file}")

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
    parser.add_argument("--parquet_file", default="sampled_segments.parquet", help="Path to the pre-sampled parquet file.")
    parser.add_argument("audio_file", help="The audio file to process.")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    myfs = do_connection(config["CONNECTION_STRING"])

    # Read the pre-sampled Parquet file into a DataFrame
    sampled_df = pd.read_parquet(args.parquet_file)

    # Normalize paths and filter by file name
    audio_file_basename = os.path.basename(args.audio_file)
    filtered_items = sampled_df[sampled_df['audio'].str.contains(audio_file_basename, na=False, case=False)]

    # Skip processing if no relevant detections are found for the file
    if filtered_items.empty:
        print(f"No detections found for {args.audio_file}. Skipping...")
        exit(0)

    # Log the number of detections
    print(f"Number of detections for {args.audio_file}: {len(filtered_items)}")

    # Process each detection
    saved_count = 0
    for _, item in filtered_items.iterrows():
        try:
            logging.info(f"Processing item: {item}")
            extract_segments(
                item,
                config["SAMPLE_RATE"],
                config["OUT_PATH_SEGMENTS"],
                myfs,
                config["CONNECTION_STRING"],
                seg_length=3,
            )
            saved_count += 1
        except Exception as e:
            logging.error(f"Error processing segment {item}: {e}")

    # Log the total number of saved segments
    print(f"Number of segments successfully saved for {args.audio_file}: {saved_count}")