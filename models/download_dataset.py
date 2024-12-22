import sys
#from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, snapshot_download

name=sys.argv[1]

save_name = 'some_dataset'
if len(sys.argv) == 3:
    save_name = sys.argv[2]

downloaded_path = snapshot_download(
    repo_id=name,
    repo_type="dataset"
)

print(downloaded_path)


#tokenizer = AutoTokenizer.from_pretrained(name)
#model = AutoModelForCausalLM.from_pretrained(name)

#model.save_pretrained(f"./{save_name}")
#tokenizer.save_pretrained(f"./{save_name}")
