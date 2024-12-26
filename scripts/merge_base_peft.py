from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_path = "/home/kashurin/.cache/huggingface/hub/models--t-tech--T-lite-it-1.0/snapshots/fbabc76f32140416dc5b0ceef392c7778eec1312/" 
lora_model_path = "/home/kashurin/.local/models/sft-grag-tlite-1.0-unsloth-lora-16-qkvogud-lr-4/checkpoint-6251/"

# Load Base Model and Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load and Merge LoRA Adapter
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload() 

# Save the merged model
save_path = "/home/kashurin/.local/models/tlite-1.0-grag-lora"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
