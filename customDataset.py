from datasets import load_dataset
import torch
import datetime
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
class CustomDatasetStage1(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512, test_run=False):
        self.dataset = load_dataset(dataset_name, split=split)
        if test_run:
            self.dataset = self.dataset.select(range(1024))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def resize_total_entries(self, new_size):
        self.dataset = self.dataset.select(range(new_size))

def tokenize_stage1(example, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"You are a helpful assistant that analyzes CUDA code for parallelization. Your task is to analyze C++ code and provide an analysis of the code. <task> ANALYZE_FOR_PARALLELIZATION </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
        },
        {
            "role": "assistant",
            "content": f"Put your final answer within <analysis>\n\boxed{example['analysis']}\n</analysis>"
        }
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_prompt = full_prompt.replace("<task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<task>"))]))
    full_prompt = full_prompt.replace("</task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</task>"))]))
    full_prompt = full_prompt.replace("<code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<code_cpp>"))]))
    full_prompt = full_prompt.replace("</code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</code_cpp>"))]))
    full_prompt = full_prompt.replace("<analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<analysis>"))]))
    full_prompt = full_prompt.replace("</analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</analysis>"))]))

    response_prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    full_enc = tokenizer(full_prompt, truncation=True, padding=True, return_tensors="pt")
    prefix_ids = tokenizer(response_prefix, truncation=True, padding=False, return_tensors="pt")["input_ids"]
    input_ids = full_enc["input_ids"]
    labels = input_ids.clone()
    masked_len = min(prefix_ids.shape[-1], labels.shape[-1])
    labels[:masked_len].fill_(-100)
    full_enc["labels"] = labels
    for k in ["input_ids", "attention_mask", "labels"]:
        if k in full_enc and isinstance(full_enc[k], torch.Tensor) and full_enc[k].dim() == 2 and full_enc[k].shape[0] == 1:
            full_enc[k] = full_enc[k].squeeze(0)
    #print("input_ids:", type(full_enc["input_ids"]), full_enc["input_ids"].shape)
    #print("labels:", type(full_enc["labels"]), full_enc["labels"].shape)
    return full_enc

def tokenize_stage2(example, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"You are a helpful assistant that analyzes and transforms C++ code to CUDA code for parallelization. Your task is to analyze C++ code and provide an analysis of the code and then transform the code to CUDA code.<task> ANALYZE_FOR_PARALLELIZATION, TRANSFORM_CODE_TO_CUDA </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
        },
        {
            "role": "assistant",
            "content": f"Put your final answer within <code_cuda>\n<kernel>\n\boxed{example['cuda_code']}\n</kernel>\n</code_cuda>"
        }
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Replace special tokens with their tokenizer string representations
    full_prompt = full_prompt.replace("<task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<task>"))]))
    full_prompt = full_prompt.replace("</task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</task>"))]))
    full_prompt = full_prompt.replace("<code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<code_cpp>"))]))
    full_prompt = full_prompt.replace("</code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</code_cpp>"))]))
    full_prompt = full_prompt.replace("<analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<analysis>"))]))
    full_prompt = full_prompt.replace("</analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</analysis>"))]))
    full_prompt = full_prompt.replace("<code_cuda>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<code_cuda>"))]))
    full_prompt = full_prompt.replace("</code_cuda>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</code_cuda>"))]))

    response_prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    full_enc = tokenizer(full_prompt, truncation=True, padding=True, return_tensors="pt")
    prefix_ids = tokenizer(response_prefix, truncation=True, padding=False, return_tensors="pt")["input_ids"]
    input_ids = full_enc["input_ids"]
    labels = input_ids.clone()
    masked_len = min(prefix_ids.shape[-1], labels.shape[-1])
    labels[:masked_len].fill_(-100)
    full_enc["labels"] = labels
    return full_enc
    
def tokenize_stage3(example, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"You are a helpful assistant that analyzes and transforms C++ code to CUDA code for parallelization. Your task is to analyze C++ code and provide an analysis of the code and then transform the code to CUDA code. <task> ANALYZE_FOR_PARALLELIZATION, TRANSFORM_CODE_TO_CUDA </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
        },
        {
            "role": "assistant",
            "content": f"Put your final answer within <code_cuda>\n\boxed{example['cuda_code']}\n</code_cuda>"
        }
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Replace special tokens with their tokenizer string representations
    full_prompt = full_prompt.replace("<task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<task>"))]))
    full_prompt = full_prompt.replace("</task>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</task>"))]))
    full_prompt = full_prompt.replace("<code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<code_cpp>"))]))
    full_prompt = full_prompt.replace("</code_cpp>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</code_cpp>"))]))
    full_prompt = full_prompt.replace("<analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<analysis>"))]))
    full_prompt = full_prompt.replace("</analysis>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</analysis>"))]))
    full_prompt = full_prompt.replace("<code_cuda>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<code_cuda>"))]))
    full_prompt = full_prompt.replace("</code_cuda>", tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("</code_cuda>"))]))

    response_prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    full_enc = tokenizer(full_prompt, truncation=True, padding=True, return_tensors="pt")
    prefix_ids = tokenizer(response_prefix, truncation=True, padding=False, return_tensors="pt")["input_ids"]
    input_ids = full_enc["input_ids"]
    labels = input_ids.clone()
    masked_len = min(prefix_ids.shape[-1], labels.shape[-1])
    labels[:masked_len].fill_(-100)
    full_enc["labels"] = labels
    return full_enc

# Tokenize the dataset for HuggingFace trainer
def tokenize_grpo(example, tokenizer):
    # Create messages in chat format
    messages = [
        {
            "role": "user",
            "content": f"<task> ANALYZE_FOR_PARALLELIZATION, TRANSFORM_CODE_TO_CUDA </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
        }
    ]
    
    # Format prompt using the tokenizer's chat template
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    # Expected response format
    response_text = f"Put your final answer within <code_cuda>\n\boxed{example['cuda_code']}\n</code_cuda>"
    
    # Tokenize with proper dimensions
    prompt_tokens = tokenizer(prompt_text, padding="max_length", max_length=2048, 
                                truncation=True, return_tensors="pt")
    
    # Ensure we have proper tensor shapes
    return {
        "prompt": prompt_text,
        "response": response_text,
        "cpp_code": example["cpp_code"],
        "cuda_code": example["cuda_code"],
        "input_ids": prompt_tokens["input_ids"][0],
        "attention_mask": prompt_tokens["attention_mask"][0]
    }

class CustomDatasetStage2(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512, test_run=False):
        self.dataset = load_dataset(dataset_name, split=split)
        if test_run:
            self.dataset = self.dataset.select(range(1024))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def resize_total_entries(self, new_size):
        self.dataset = self.dataset.select(range(new_size))
