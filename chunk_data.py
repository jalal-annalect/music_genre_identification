import os
from utils import chunk_audio
# read from dir
train_root_dir = "dataset_30sec_audio/train"
val_root_dir = "dataset_30sec_audio/val"

# creat output dirs
# make train and val folders
os.mkdir("dataset_3sec_audio")
os.mkdir("dataset_3sec_audio/train")
os.mkdir("dataset_3sec_audio/val")

# output dirs
out_train_root_dir = "dataset_3sec_audio/train"
out_val_root_dir = "dataset_3sec_audio/val"

# make sure to only take audio files
train_files = [elem for elem in os.listdir(train_root_dir) if elem.endswith(".wav")]
val_files = [elem for elem in os.listdir(val_root_dir) if elem.endswith(".wav")]

# splitting training data
for elem in train_files:
    audio_path = os.path.join(train_root_dir, elem)
    chunk_audio(filename=audio_path, output_dir=out_train_root_dir, window=3)

# splitting valing data
for elem in val_files:
    audio_path = os.path.join(val_root_dir, elem)
    chunk_audio(filename=audio_path, output_dir=out_val_root_dir, window=3)