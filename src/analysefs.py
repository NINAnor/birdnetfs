import datetime
import operator
import os
import pathlib
import sys
import config as cfg
from birdnetsrc.analyze import (
    get_result_file_names,
    getSortedTimestamps,
    loadCodes,
    predict,
    #saveResultFile
)
from birdnetsrc.audio import splitSignal
from birdnetsrc.utils import readLines, writeErrorLog, save_result_file
from utils import read_audio_data

RTABLE_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"
RAVEN_TABLE_HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"

def generate_raven_table(timestamps: list[str], result: dict[str, list], afile_path: str, result_path: str, sample_rate: int) -> str:
    selection_id = 0
    out_string = RAVEN_TABLE_HEADER

    # Read native sample rate
    high_freq = sample_rate / 2

    if high_freq > cfg.SIG_FMAX:
        high_freq = cfg.SIG_FMAX

    high_freq = min(high_freq, cfg.BANDPASS_FMAX)
    low_freq = max(cfg.SIG_FMIN, cfg.BANDPASS_FMIN)

    # Extract valid predictions for every timestamp
    for timestamp in timestamps:
        rstring = ""
        start, end = timestamp.split("-", 1)

        for c in result[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                selection_id += 1
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                code = cfg.CODES[c[0]] if c[0] in cfg.CODES else c[0]
                rstring += f"{selection_id}\tSpectrogram 1\t1\t{start}\t{end}\t{low_freq}\t{high_freq}\t{label.split('_', 1)[-1]}\t{code}\t{c[1]:.4f}\t{afile_path}\t{start}\n"

        # Write result string to file
        out_string += rstring

    # If we don't have any valid predictions, we still need to add a line to the selection table in case we want to combine results
    # TODO: That's a weird way to do it, but it works for now. It would be better to keep track of file durations during the analysis.
    if len(out_string) == len(RAVEN_TABLE_HEADER) and cfg.OUTPUT_PATH is not None:
        selection_id += 1
        out_string += (
            f"{selection_id}\tSpectrogram 1\t1\t0\t3\t{low_freq}\t{high_freq}\tnocall\tnocall\t1.0\t{afile_path}\t0\n"
        )
        
    save_result_file(result_path, out_string)

def saveResultFiles(r: dict[str, list], result_files: dict[str, str], afile_path: str, sample_rate: int):
    """Saves the results to the hard drive.

    Args:
        r: The dictionary with {segment: scores}.
        path: The path where the result should be saved.
        afile_path: The path to audio file.
    """

    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    # Selection table
    timestamps = getSortedTimestamps(r)

    if "table" in result_files:
        generate_raven_table(timestamps, r, afile_path, result_files["table"], sample_rate)


def analyzeFile(fpath: pathlib.Path):
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
    result_file_name = get_result_file_names(fpath)

    # Open file and split into chunks:
    wave, sr, fileLengthSeconds = read_audio_data(fpath, sr=cfg.SAMPLE_RATE)

    # Status
    print(f"Analyzing {fpath}", flush=True)

    # Process each chunk
    while offset < fileLengthSeconds:
        chunks = splitSignal(
            wave, sr, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN
        )
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
                p_labels = zip(cfg.LABELS, pred, strict=False)

                # Sort by score
                p_sorted = sorted(
                    p_labels, key=operator.itemgetter(1), reverse=True
                )

                # Store top 5 results and advance indices
                results[str(s_start) + "-" + str(s_end)] = p_sorted

            # Clear batch
            samples = []
            timestamps = []
        offset = offset + duration

    print(f"FPATH IS {fpath}")
    saveResultFiles(results, result_file_name, fpath, cfg.SAMPLE_RATE)

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Finished {fpath} in {delta_time:.2f} seconds", flush=True)

    return True


if __name__ == "__main__":
    # Set paths relative to script path (requested in #3)
    script_dir = pathlib.Path(sys.argv[0]).parent.absolute() / "birdnetsrc"
    cfg.MODEL_PATH = os.path.join(script_dir, cfg.MODEL_PATH)
    cfg.LABELS_FILE = script_dir / cfg.LABELS_FILE
    cfg.TRANSLATED_LABELS_PATH = script_dir / cfg.TRANSLATED_LABELS_PATH
    cfg.MDATA_MODEL_PATH = script_dir / cfg.MDATA_MODEL_PATH
    cfg.CODES_FILE = script_dir / cfg.CODES_FILE
    cfg.ERROR_LOG_FILE = pathlib.Path() / cfg.ERROR_LOG_FILE

    # Load eBird codes, labels
    cfg.CODES = loadCodes()
    cfg.LABELS = readLines(cfg.LABELS_FILE)

    cfg.TRANSLATED_LABELS = cfg.LABELS

    cfg.SPECIES_LIST_FILE = pathlib.Path() / "species_list.txt"
    cfg.SPECIES_LIST = readLines(cfg.SPECIES_LIST_FILE)

    filename = sys.argv[1]
    analyzeFile(filename)
