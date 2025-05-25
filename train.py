from utils import *
import argparse
import torch
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from customDataset import *
from model import *
import safetensors.torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_path', type=str, default='./model_config/config.yaml', help='Path to the config file')
parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
args = parser.parse_args()

config = read_config(args.config_path)
print(config)
model_name = config["base_model"]["model_name"]
stage1 = config["stage_config"]["stages"]["stage1"]
stage2 = config["stage_config"]["stages"]["stage2"]
stage3 = config["stage_config"]["stages"]["stage3"]
stage4 = config["stage_config"]["stages"]["stage4"]
eval = config["stage_config"]["stages"]["eval"]
test_run = config["dataset"]["test_run"]

if stage1:
    decoder1, decoder2, tokenizer, hidden_dim, model_config = load_model_and_tokenizer(model_name, special_tokens=True)
elif stage2:
    decoder2, tokenizer, hidden_dim, model_config = load_model_and_tokenizer_stage2(model_name, special_tokens=True)

if stage1 or stage3 or stage4:
    ds_stage1 = CustomDatasetStage1(
        dataset_name=config["dataset"]["dataset_hub_path"],
        split=config["dataset"]["split"],
        tokenizer=tokenizer,
        max_length=config["dataset"]["max_length"],
        test_run=test_run
    )

if stage1:
    print("Stage 1: Training stage 1 decoder1")
    wandb.init(
        project="accfiy_test",
        name = f"Stage1_mv2_Experiment_{dt}",
    )
    wandb.run.name = "stage1_full_dataset"
    hf_dataset_stage1 = ds_stage1.dataset
    hf_dataset_stage1 = hf_dataset_stage1.map(tokenize_stage1, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
    print(f"Dataset train :{hf_dataset_stage1['input_ids'].__class__}")
    print(f"Dataset attention :{hf_dataset_stage1['attention_mask'].__class__}")
    hf_dataset_stage1 = hf_dataset_stage1.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    # Clear CUDA cache before training
    torch.cuda.empty_cache()
    
    Train_stage1(
        model=decoder1,
        train_ds=hf_dataset_stage1["train"],
        eval_ds=hf_dataset_stage1["test"],
        tokenizer=tokenizer,
    )
    wandb.finish()
    
    # Clear CUDA cache after training
    torch.cuda.empty_cache()

# Load the trained decoder1 model
decoder1 = AutoModelForCausalLM.from_pretrained("./output_decoder1_test/")
combined_model = CombinedModel(model_config, model_name, decoder1, decoder2, hidden_dim, tokenizer=tokenizer)

# Freeze decoder1
for param in combined_model.decoder1.parameters():
    param.requires_grad = False

# (Just to be explicit, but not strictly needed)
for param in combined_model.decoder2.parameters():
    param.requires_grad = True

for param in combined_model.mapper.parameters():
    param.requires_grad = True
combined_model.to(device)

if stage2:
    print("Stage 2: Training combined model with second synthetic dataset")
    # wandb.init(
    #     project="accfiy_test",
    #     name = f"Stage2_mv2_Experiment_{dt}",
    # )
    # wandb.run.name = "stage2_full_dataset"
    # Load the second synthetic dataset
    ds_stage2 = CustomDatasetStage2(
        dataset_name=config["dataset2"]["dataset_hub_path"],
        split=config["dataset2"]["split"],
        tokenizer=tokenizer,
        max_length=config["dataset2"]["max_length"],
        test_run=test_run
    )
    hf_dataset_stage2 = ds_stage2.dataset
    hf_dataset_stage2 = hf_dataset_stage2.map(tokenize_stage2, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
    hf_dataset_stage2 = hf_dataset_stage2.train_test_split(test_size=0.1, shuffle=True, seed=42)
    Train_stage2(
        model=combined_model,
        train_ds=hf_dataset_stage2["train"],
        eval_ds=hf_dataset_stage2["test"],
        tokenizer=tokenizer,
    )
    # wandb.finish()

if stage3:
    print("Stage 3: Training combined model")
    wandb.init(
        project="accfiy_test",
        name = f"Stage3Experiment_{dt}",
    )
    wandb.run.name = "stage3_full_dataset"
    decoder1 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage2_test/decoder1/")
    decoder2 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage2_test/decoder2/")
    mapper_state = torch.load("./output_combined_model_stage2_test/mapper.pt", map_location=device)
    tokenizer = AutoTokenizer.from_pretrained("./output_combined_model_stage2_test/")
    #print(f"Tokenizer loaded successfully: {tokenizer}")
    combined_model = CombinedModel(model_config, model_name, decoder1, decoder2, hidden_dim, mapper_state=mapper_state, tokenizer=tokenizer)
    #combined_model.to(device)
    hf_dataset_stage3 = ds_stage1.dataset
    hf_dataset_stage3 = hf_dataset_stage3.map(tokenize_stage3, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
    hf_dataset_stage3 = hf_dataset_stage3.train_test_split(test_size=0.1, shuffle=True, seed=42)
    Train_stage3(
        model=combined_model,
        train_ds=hf_dataset_stage3["train"],
        eval_ds=hf_dataset_stage3["test"],
        tokenizer=tokenizer,
    )
    wandb.finish()
    
if stage4:
    print("Stage 4: Training combined model")
    wandb.init(
        project="accfiy_test",
        name = f"Stage4Experiment_{dt}",
    )
    wandb.run.name = "stage4_full_dataset"
    decoder1 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage3/decoder1/")
    decoder2 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage3/decoder2/")
    mapper_state = torch.load("./output_combined_model_stage3/mapper.pt", map_location=device)
    combined_model = CombinedModel(model_config, model_name, decoder1, decoder2, hidden_dim, mapper_state=mapper_state)
    tokenizer = AutoTokenizer.from_pretrained("./output_combined_model_stage3/")
    print("Combined model loaded successfully")
    hf_dataset_stage4 = ds_stage1.dataset
    hf_dataset_stage4 = hf_dataset_stage4.map(tokenize_grpo, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)
    hf_dataset_stage4 = hf_dataset_stage4.train_test_split(test_size=0.1, shuffle=True, seed=42)
    Train_stage4(
        model=combined_model,
        train_ds=hf_dataset_stage4["train"],
        eval_ds=hf_dataset_stage4["test"],
        tokenizer=tokenizer,
    )
    wandb.finish()
    

if eval:
    print("Stage Eval: Evaluating the model")
    from datasets import load_dataset, Dataset
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    from evaluate import load as load_metric
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load evaluation dataset
    eval_dataset = pd.read_json("cpp_cuda_eval_dataset.json")
    dataset = eval_dataset.to_dict(orient="records")

    # # Load model and tokenizer
    # model_name = "Qwen/Qwen3-1.7B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load the trained combined model and tokenizer
    decoder1 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage2_test/decoder1/")
    decoder2 = AutoModelForCausalLM.from_pretrained("./output_combined_model_stage2_test/decoder2/")
    mapper_state = torch.load("./output_combined_model_stage2_test/mapper.pt", map_location=device)
    combined_model = CombinedModel(model_config, model_name, decoder1, decoder2, hidden_dim, mapper_state=mapper_state)
    tokenizer = AutoTokenizer.from_pretrained("./output_combined_model_stage2_test/")
    combined_model.to(device)
    print("Combined model loaded successfully")

    # Evaluate model
    def evaluate_model(dataset, model, tokenizer, max_new_tokens=2048):
        predictions = []
        for item in tqdm(dataset):
            # prompt = f"{item['prompt']}\n{item['cpp_code']}\n"
            # inputs = tokenizer(prompt, return_tensors="pt").to(device)
            messages = [
                {
                    "role": "user",
                    "content": f"<task> ANALYZE_FOR_PARALLELIZATION, TRANSFORM_CODE_TO_CUDA </task>\n<code_cpp>\n{item['cpp_code']}\n</code_cpp>"
                }
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=2048, return_tensors="pt")
            inputs = inputs.to(device)

            outputs = model.generate(
                **inputs,
                # inputs_ids=inputs["input_ids"],
                # attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(decoded.split(item['cpp_code'])[-1].strip())
            predictions.append(decoded.split(item['cpp_code'])[-1].strip())
        return predictions

    preds = evaluate_model(dataset, combined_model, tokenizer)

    # Compute BLEU
    bleu = load_metric("bleu")
    references = [x['expected_output'] for x in dataset]
    results = bleu.compute(predictions=preds, references=references)
    print("BLEU Score:", results['bleu'])