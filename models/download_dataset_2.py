import os
import sys
from datasets import load_dataset

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

name='Vikhrmodels/Grounded-RAG-RU-v2'

save_name = 'some_dataset'

ds = load_dataset(name)

ds.save_to_disk(save_name)
