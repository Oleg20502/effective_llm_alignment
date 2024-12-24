import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

name='unsloth/Meta-Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

ds_path='Vikhrmodels/Grounded-RAG-RU-v2'

ds = load_dataset(ds_path) 
