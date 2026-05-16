import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle_imports import run_oracle_single_layer, run_oracle_multi_layer, visualize_token_selection

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

# Configuration for the Llama EM model
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EM_LORA_PATH = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
LLAMA_ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"

load_dotenv(dotenv_path=str(".env"))
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN in your chapter1_transformer_interp/exercises/.env file"


MAIN = __name__ == "__main__"

if MAIN: 
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    # Add dummy adapter for consistent PeftModel API
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    print("Model loaded successfully!")

    print(f"Loading oracle LoRA: {ORACLE_LORA_PATH}")
    model.load_adapter(ORACLE_LORA_PATH, adapter_name="oracle", is_trainable=False)
    print("Oracle loaded successfully!")

    config_dict = model.peft_config["oracle"].to_dict()
    config_df = pd.DataFrame(list(config_dict.items()), columns=["Parameter", "Value"])