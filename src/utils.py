import glob
import os
import shutil
import tempfile

import audioread
import fsspec
import librosa


def read_file(filepath, sr):
    # Step 1: Create a temporary directory for this process
    # temp_dir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    # print(f"Created temp directory: {temp_dir}")

    # Step 2: Use fsspec and configure it to use the temp_dir for caching
    with fsspec.open(
        filepath
    ) as f:  #     with fsspec.open(filepath, block_cache_dir=temp_dir) as f:
        wave, fs = librosa.load(f, sr=sr, mono=True, res_type="kaiser_fast")

    # Return the data and the temp directory path (for later cleanup)
    return (
        wave,
        fs,
    )  # temp_dir


def clean_tmp(temp_dir):
    # Step 3: Clean up the temporary directory after use
    try:
        shutil.rmtree(temp_dir)
        print(f"Deleted temp directory: {temp_dir}")
    except Exception as e:
        print(f"Failed to delete temp directory {temp_dir}. Reason: {e}")


def read_audio_data(path, sr):
    try:
        ndarray, rate = read_file(path, sr)  # , tmpdir
        duration = librosa.get_duration(y=ndarray, sr=sr)
    except audioread.exceptions.NoBackendError as e:
        print(e)
    return ndarray, rate, duration  # , tmpdir


def openCachedFile(filesystem, path, sample_rate=48000, offset=0.0, duration=None):
    import shutil

    bin = filesystem.openbin(path)

    with tempfile.NamedTemporaryFile() as temp:
        shutil.copyfileobj(bin, temp)
        sig, rate = openAudioFile(temp.name, sample_rate, offset, duration)

    return sig, rate


def openAudioFile(path, sample_rate=44100, offset=0.0, duration=None):
    try:
        sig, rate = librosa.load(
            path,
            sr=sample_rate,
            offset=offset,
            duration=duration,
            mono=True,
            res_type="kaiser_fast",
        )
    except:
        sig, rate = [], sample_rate

    return sig, rate


def saveSignal(sig, fname):
    import soundfile as sf

    sf.write(fname, sig, 48000, "PCM_16")


#####################################################################
######################### PARSING UTILS #############################
#####################################################################


def remove_extension(input):
    filename = input.split("/")[-1].split(".")[0]
    if len(filename) > 2:
        filename = ".".join(filename[0:-1])
    else:
        filename = input.split("/")[-1].split(".")[0]
    return filename


def parseFolders(apath, rpath):
    audio_files = [
        f for f in glob.glob(apath + "/**/*", recursive=True) if os.path.isfile(f)
    ]
    audio_no_extension = []
    for audio_file in audio_files:
        audio_file_no_extension = remove_extension(audio_file)
        audio_no_extension.append(audio_file_no_extension)

    result_files = [
        f for f in glob.glob(rpath + "/**/*", recursive=True) if os.path.isfile(f)
    ]

    flist = []
    for result in result_files:
        result_no_extension = remove_extension(result)
        is_in = result_no_extension in audio_no_extension

        if is_in:
            audio_idx = audio_no_extension.index(result_no_extension)
            pair = {"audio": audio_files[audio_idx], "result": result}
            flist.append(pair)
        else:
            continue

    print(f"Found {len(flist)} audio files with valid result file.")

    return flist
