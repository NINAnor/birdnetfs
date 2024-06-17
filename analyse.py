import datetime
import operator
import sys
import numpy as np
import os

import config as cfg
import birdnetsrc.model
from birdnetsrc.analyze import predict, get_result_file_name, saveResultFile, getSortedTimestamps
from birdnetsrc.audio import splitSignal
from birdnetsrc.utils import writeErrorLog

from utils import read_audio_data

RTABLE_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"

def saveResultFile(r: dict[str, list], path: str, afile_path: str, sample_rate: int):
    """Saves the results to the hard drive.

    Args:
        r: The dictionary with {segment: scores}.
        path: The path where the result should be saved.
        afile_path: The path to audio file.
    """
    # Make folder if it doesn't exist
    print(f"PATH IS {path}")
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Selection table
    out_string = ""

    if cfg.RESULT_TYPE == "table":
        selection_id = 0
        filename = os.path.basename(afile_path)

        # Write header
        out_string += RTABLE_HEADER

        # Read native sample rate
        high_freq = sample_rate / 2

        if high_freq > cfg.SIG_FMAX:
            high_freq = cfg.SIG_FMAX

        high_freq = min(high_freq, cfg.BANDPASS_FMAX)
        low_freq = max(cfg.SIG_FMIN, cfg.BANDPASS_FMIN)

        # Extract valid predictions for every timestamp
        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    selection_id += 1
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    code = cfg.CODES[c[0]] if c[0] in cfg.CODES else c[0]
                    rstring += f"{selection_id}\tSpectrogram 1\t1\t{start}\t{end}\t{low_freq}\t{high_freq}\t{label.split('_', 1)[-1]}\t{code}\t{c[1]:.4f}\t{afile_path}\t{start}\n"

            # Write result string to file
            out_string += rstring

        # If we don't have any valid predictions, we still need to add a line to the selection table in case we want to combine results
        # TODO: That's a weird way to do it, but it works for now. It would be better to keep track of file durations during the analysis.
        if len(out_string) == len(RTABLE_HEADER) and cfg.OUTPUT_PATH is not None:
            selection_id += 1
            out_string += f"{selection_id}\tSpectrogram 1\t1\t0\t3\t{low_freq}\t{high_freq}\tnocall\tnocall\t1.0\t{afile_path}\t0\n"

    elif cfg.RESULT_TYPE == "audacity":
        # Audacity timeline labels
        for timestamp in getSortedTimestamps(r):
            rstring = ""

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    ts = timestamp.replace("-", "\t")
                    lbl = label.replace("_", ", ")
                    rstring += f"{ts}\t{lbl}\t{c[1]:.4f}\n"

            # Write result string to file
            out_string += rstring

    elif cfg.RESULT_TYPE == "r":
        # Output format for R
        header = "filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model"
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "\n{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}".format(
                        afile_path,
                        start,
                        end,
                        label.split("_", 1)[0],
                        label.split("_", 1)[-1],
                        c[1],
                        cfg.LATITUDE,
                        cfg.LONGITUDE,
                        cfg.WEEK,
                        cfg.SIG_OVERLAP,
                        (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
                        cfg.MIN_CONFIDENCE,
                        cfg.SPECIES_LIST_FILE,
                        os.path.basename(cfg.MODEL_PATH),
                    )

            # Write result string to file
            out_string += rstring

    elif cfg.RESULT_TYPE == "kaleidoscope":
        # Output format for kaleidoscope
        header = "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity"
        out_string += header

        folder_path, filename = os.path.split(afile_path)
        parent_folder, folder_name = os.path.split(folder_path)

        for timestamp in getSortedTimestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "\n{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{},{}".format(
                        parent_folder.rstrip("/"),
                        folder_name,
                        filename,
                        start,
                        float(end) - float(start),
                        label.split("_", 1)[0],
                        label.split("_", 1)[-1],
                        c[1],
                        cfg.LATITUDE,
                        cfg.LONGITUDE,
                        cfg.WEEK,
                        cfg.SIG_OVERLAP,
                        (1.0 - cfg.SIGMOID_SENSITIVITY) + 1.0,
                    )

            # Write result string to file
            out_string += rstring

    else:
        # CSV output file
        header = "Start (s),End (s),Scientific name,Common name,Confidence\n"

        # Write header
        out_string += header

        for timestamp in getSortedTimestamps(r):
            rstring = ""

            for c in r[timestamp]:
                start, end = timestamp.split("-", 1)

                if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                    label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                    rstring += "{},{},{},{},{:.4f}\n".format(
                        start, end, label.split("_", 1)[0], label.split("_", 1)[-1], c[1]
                    )

            # Write result string to file
            out_string += rstring

    # Save as file
    with open(path, "w", encoding="utf-8") as rfile:
        rfile.write(out_string)

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
    wave, sr, fileDuration = read_audio_data(fpath, sr=cfg.SAMPLE_RATE)

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    try:
        while offset < fileDuration:
            chunks = splitSignal(wave, sr, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
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
        print(f"RESULT FILE NAME IS {result_file_name}")
        print(f"FPATH IS {fpath}")
        saveResultFile(results, result_file_name, fpath, cfg.SAMPLE_RATE)

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
    analyzeFile(filename)