import os
import shutil
import fsspec
import librosa
import audioread

def read_file(filepath, sr):
    with fsspec.open(filepath) as f:
        wave, fs = librosa.load(f,sr=sr, mono=True, res_type="kaiser_fast")
    return wave, fs

def read_audio_data(path, sr):
    try:
        ndarray, rate = read_file(path, sr)
        duration = librosa.get_duration(y=ndarray, sr=sr)
    except audioread.exceptions.NoBackendError as e:
        print(e)
    return ndarray, rate, duration

def clean_tmp(directory="/tmp"):
    # Target only directories starting with 'tmp'
    for folder_name in os.listdir(directory):
        if folder_name.startswith("tmp"):
            folder_path = os.path.join(directory, folder_name)
            try:
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
            except Exception as e:
                print(f'Failed to delete {folder_path}. Reason: {e}')
