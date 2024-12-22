import os
import sys
from datasets import load_dataset

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ds = load_dataset('/home/kashurin/.local/Grounded-RAG-RU-v2')
