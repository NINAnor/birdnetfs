import datetime
import operator
import sys

import config as cfg
from birdnetsrc.analyze import predict, get_result_file_name, saveResultFile
from birdnetsrc.audio import splitSignal
from birdnetsrc.utils import writeErrorLog

from utils import read_audio_data

def analyzeFile(fpath: str):
    """Analyzes a file.

    Predicts the scores for the file and saves the results.

    Args:
        item: Tuple containing (file path, config)

    Returns:
        The `True` if the file was analyzed successfully.
    """

    # Start time
    start_time = datetime.datetime.now()
    offset = 0
    duration = cfg.FILE_SPLITTING_DURATION
    start, end = 0, cfg.SIG_LENGTH
    results = {}
    result_file_name = get_result_file_name(fpath)

    # Open file and split into chunks:
    wave, fileDuration, sr = read_audio_data(fpath)

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        while offset < fileDuration:
            chunks = splitSignal(wave, sr, cfg.SIG_LENGTH)
            samples = []
            timestamps = []

            for chunk_index, chunk in enumerate(chunks):
                # Add to batch
                samples.append(chunk)
                timestamps.append([start, end])

                # Advance start and end
                start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                end = start + cfg.SIG_LENGTH

                # Check if batch is full or last chunk
                if len(samples) < cfg.BATCH_SIZE and chunk_index < len(chunks) - 1:
                    continue

                # Predict
                p = predict(samples)
                print(p)
                # Add to results
                for i in range(len(samples)):
                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Get prediction
                    pred = p[i]

                    # Assign scores to labels
                    p_labels = zip(cfg.LABELS, pred)

                    # Sort by score
                    p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)

                    # Store top 5 results and advance indices
                    results[str(s_start) + "-" + str(s_end)] = p_sorted

                # Clear batch
                samples = []
                timestamps = []
            offset = offset + duration

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot analyze audio file {fpath}.\n", flush=True)
        writeErrorLog(ex)

        return False

    # Save as selection table
    try:
        saveResultFile(results, result_file_name, fpath)

    except Exception as ex:
        # Write error log
        print(f"Error: Cannot save result for {fpath}.\n", flush=True)
        writeErrorLog(ex)

        return False

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)

    return True

if __name__ == "__main__":

    filename = sys.argv[1]
    print(filename)
    analyzeFile(filename)