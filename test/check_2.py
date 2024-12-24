import os
import sys
from datasets import load_dataset

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

dir='/home/kashurin/.cache/huggingface/hub/datasets--Vikhrmodels--Grounded-RAG-RU-v2/snapshots/7be9767db044c5b3f460557356ccd4cd8e69dfe1/data/'

files = [dir+'train-00000-of-00002.parquet', dir+'train-00001-of-00002.parquet', dir+'test-00000-of-00001.parquet']

for path in files:
    dataset = load_dataset(path.split('.')[-1], data_files=path)
