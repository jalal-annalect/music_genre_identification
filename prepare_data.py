import librosa
import os
import warnings
import utils
import random
import shutil
import integv
warnings.filterwarnings("ignore")

root_dir = "genres_original"

# remove corrupted files
for elem in os.listdir(root_dir):
    dir = f"{root_dir}/{elem}"
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            corrupt = not integv.verify(open(f"{dir}/{file}", "rb"), file_type="wav")
            if not corrupt:
                pass
            else:
                print(f"Found corrupt file, will remove {dir}/{file}.")
                os.remove(f"{dir}/{file}")

# make train and val folders
os.mkdir("dataset_30sec_audio")
os.mkdir("dataset_30sec_audio/train")
os.mkdir("dataset_30sec_audio/val")

# creating train and val data
for elem in os.listdir(root_dir):

    dir = f"{root_dir}/{elem}"
    if os.path.isdir(dir):
        files = os.listdir(dir)
        print(f"found: {len(files)} files in category {elem}")

        # shuffle files
        random.shuffle(files)
        random.shuffle(files)

        # splitting
        train_files = files[:80]
        val_files = files[80:]

        # copying train files
        for train_file in train_files:
            if os.path.isfile(f"{dir}/{train_file}"):
                shutil.copy(f"{dir}/{train_file}", f"dataset_30sec_audio/train/{train_file}")
            else:
                raise Exception(f"""{dir}/{train_file} not found""")
                
        # copying val files
        for val_file in val_files:
            if os.path.isfile(f"{dir}/{val_file}"):
                shutil.copy(f"{dir}/{val_file}", f"dataset_30sec_audio/val/{val_file}")
            else:
                raise Exception(f"""{dir}/{val_file} not found""")
    else:
        pass