from transformers import BitsAndBytesConfig
import torch

HF_BASE_LINK = "https://router.huggingface.co/v1"
DATA_LINK = "https://raw.githubusercontent.com/allenkong221/netflix-titles-dataset/master/netflix_titles.csv"
DATA_LINK_IMDB = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/imdb_1000.csv"
QUANT_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
QWEN_PATH = "Qwen/Qwen2.5-3B-Instruct"
LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"