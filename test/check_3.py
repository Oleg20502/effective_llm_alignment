import os
import sys
from datasets import load_from_disk

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ds = load_from_disk('some_dataset') 
