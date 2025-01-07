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
    parser.add_argument("--parquet_file", default="sample.parquet")
    parser.add_argument("audio_file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    myfs = do_connection(config["CONNECTION_STRING"])

    # Read the Parquet file into a DataFrame
    parquet_df = pd.read_parquet(args.parquet_file)

    # Log the total number of detections in the Parquet file
    print(f"Total number of detections in the Parquet file: {len(parquet_df)}")

    # Limit the total number of segments per species globally
    num_segments_per_species = config["NUM_SEGMENTS"]
    sampled_items = parquet_df.groupby("species").apply(
        lambda x: x.sample(min(len(x), num_segments_per_species), random_state=None)
    ).reset_index(drop=True)

    # Log the number of sampled detections
    print(f"Number of sampled detections across all species: {len(sampled_items)}")

    # Filter the sampled items for the specific audio file
    audio_file_basename = os.path.basename(args.audio_file)
    filtered_items = sampled_items[sampled_items['audio'].str.contains(audio_file_basename, na=False, case=False)]

    # Skip processing if there are no relevant segments for the file
    if filtered_items.empty:
        print(f"No detections found for {args.audio_file}. Skipping...")
        exit(0)

    # Log the number of detections for the specific audio file
    print(f"Number of detections for {args.audio_file}: {len(filtered_items)}")

    # Process each detection
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
        except Exception as e:
            logging.error(f"Error processing segment {item}: {e}")