import multiprocessing
import os
import random
import uuid
import warnings
import logging
from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig, ModelConfig, get_peft_config

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.configs.common_script_args import CommonScriptArguments
from src.utils.datasets import load_datasets
from src.utils.logger import setup_logging
from src.utils.model_preparation import setup_model_and_tokenizer
from src.utils.yaml_args_parser import H4ArgumentParser

logger = get_logger(__name__)

logging.basicConfig(level=logging.INFO)
train_logger = logging.getLogger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            train_logger.info(f"Step {state.global_step}: Loss = {logs['loss']}")

class PrepareForInferenceCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        FastLanguageModel.for_inference(model)


@dataclass
class SFTScriptArguments(CommonScriptArguments):
    conversation_field: str | None = field(
        default="prompt",
        metadata={"help": "Field in dataset with conversations (in list of dicts format)"}
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Will use system prompt if there is no one in dialogue, set to None to disable"}
    )
    train_only_on_completions: bool | None = field(
        default=True,
        metadata={"help": "Do train only on completions or not"}
    )
    generate_eval_examples: bool | None = field(
        default=True,
        metadata={"help": "Do generate examples on eval"}
    )
    assistant_message_template: str | None = field(
        default="<|start_header_id|>assistant<|end_header_id|>\n\n",
        metadata={"help": "Assistant message template for the training only on completions"}
    )
    num_gen_examples: int | None = field(
        default=50,
        metadata={"help": "Number of examples to generate on eval phase"}
    )
    model_support_system_role: bool | None = field(
        default=True,
        metadata={"help": "Flag that indicates if model have support for system prompt. If not, will use user for setting system prompt"}
    )

    def __post_init__(self):
        self.project_name = "sft-tuning" if self.project_name == "default-project" else self.project_name


def main():
    parser = H4ArgumentParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, sft_config, model_config = parser.parse()

    setup_logging(logger, sft_config)
    set_seed(sft_config.seed)  # in case of new tokens added without initialize...

    ################
    # Model & Tokenizer
    ################
    
    print(f'\n\n\nargs')
    print(f'{args}\n\n\n')

    print(f'\n\n\nsft_config')
    print(f'{sft_config}\n\n\n')

    print(f'\n\n\nmodel_config')
    print(f'{model_config}\n\n\n')


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name_or_path,
        max_seq_length=sft_config.max_seq_length,
        dtype=torch.bfloat16 if sft_config.bf16 else torch.float16,
        load_in_4bit=True,
	    local_files_only=True
    )

    print(f'\n\n\n')
    print(f'tokenizer.model_max_length {tokenizer.model_max_length}')
    print(f'tokenizer.bos_token {tokenizer.bos_token} {tokenizer.bos_token_id}')
    print(f'tokenizer.eos_token {tokenizer.eos_token} {tokenizer.eos_token_id}')
    print(f'tokenizer.pad_token {tokenizer.pad_token} {tokenizer.pad_token_id}')
    print(f'tokenizer.chat_template {tokenizer.chat_template}')
    print(f'\n\n\n')



    model = FastLanguageModel.get_peft_model(
        model,
        r = model_config.lora_r,
        target_modules = model_config.lora_target_modules,
        lora_alpha = model_config.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = sft_config.seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    setup_model_and_tokenizer(args, model, tokenizer, sft_config.max_seq_length)

    print(f'\n\n\n')
    print(f'tokenizer.model_max_length {tokenizer.model_max_length}')
    print(f'tokenizer.bos_token {tokenizer.bos_token} {tokenizer.bos_token_id}')
    print(f'tokenizer.eos_token {tokenizer.eos_token} {tokenizer.eos_token_id}')
    print(f'tokenizer.pad_token {tokenizer.pad_token} {tokenizer.pad_token_id}')
    print(f'tokenizer.chat_template {tokenizer.chat_template}')
    print(f'\n\n\n')    


if __name__ == '__main__':
    main()
