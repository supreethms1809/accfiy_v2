import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, AutoConfig, AutoModel, GenerationMixin
import datetime
import os
import json

dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Load the model and tokenizer
def load_model_and_tokenizer(model_name, special_tokens=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    #decoder1 = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
    #decoder2 = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
    #attn_implementation = "eager"
    attn_implementation = "flash_attention_2"
    decoder1 = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation=attn_implementation, torch_dtype=torch.float16)
    decoder2 = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation=attn_implementation, torch_dtype=torch.float16)
    print(f"Successfully loaded model")
    model_config = decoder1.config
    hidden_dim = model_config.hidden_size
    if special_tokens:
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
        decoder1.resize_token_embeddings(len(tokenizer))
        decoder2.resize_token_embeddings(len(tokenizer))
    decoder1.config.use_cache = False
    decoder2.config.use_cache = False
    decoder1.attn_implementation = attn_implementation
    decoder2.attn_implementation = attn_implementation
    return decoder1, decoder2, tokenizer, hidden_dim, model_config

# Load the model and tokenizer
def load_model_and_tokenizer_stage2(model_name, special_tokens=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    attn_implementation = "flash_attention_2"
    decoder2 = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation=attn_implementation)
    model_config = decoder2.config
    hidden_dim = model_config.hidden_size
    if special_tokens:
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
        #decoder1.resize_token_embeddings(len(tokenizer))
        decoder2.resize_token_embeddings(len(tokenizer))
    #decoder1.config.use_cache = False
    decoder2.config.use_cache = False
    #decoder1.attn_implementation = "eager"
    decoder2.attn_implementation = attn_implementation
    return decoder2, tokenizer, hidden_dim, model_config

# --- CombinedModel for TRL GRPOTrainer ---
class CombinedModel(PreTrainedModel, GenerationMixin):
    def __init__(self, model_config, model_name, decoder1, decoder2, hidden_dim, **kwargs):
        model_config.attn_implementation = "flash_attention_2"
        super().__init__(model_config)
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        if (mapper_state := kwargs.get("mapper", None)) is not None:
            self.mapper = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.mapper.load_state_dict(mapper_state)
        else:
            self.mapper = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config._name_or_path = model_name
        self.config.hidden_size = hidden_dim
        self.config.attn_implementation = "flash_attention_2"
        self.config.use_cache = False
        self.mapper_gradient_checkpointing = False
        self.warnings_issued = {}
        self.add_model_tags = lambda *args, **kwargs: None
        self.tokenizer = kwargs.get("tokenizer", None)
        self.max_position_embeddings = self.config.max_position_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids.dim() == 3 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask.squeeze(1)
        # if hasattr(self.decoder1.generate, '__wrapped__'):
        #     self.decoder1.generate = self.decoder1.generate.__wrapped__.__get__(self.decoder1, type(self.decoder1))
        with torch.no_grad():
            outputs = self.decoder1.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_position_embeddings,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                use_cache=False,
            )
            
        # Create attention mask for the generated sequence
        gen_attention_mask = torch.ones_like(outputs)
        
        # # # Print the actual text
        # input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # print("\nInput text:")
        # print(input_text)
        # print("\nGenerated text:")
        # print(generated_text)
        # print("\nNewly generated part:")
        # print(generated_text[len(input_text):])
        
        # # # Get only the last hidden state and immediately clean up outputs
        # # last_hidden = outputs.hidden_states[-1][-1]
        
        # # # Get only the generated portion of hidden states
        # # gen_hidden = last_hidden[:, input_ids.shape[1]:, :]
        # # del last_hidden
        # # torch.cuda.empty_cache()
        output_decoder1_pass = self.decoder1(
            input_ids=outputs,
            attention_mask=gen_attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        gen_hidden = output_decoder1_pass.hidden_states[-1][:, input_ids.shape[1]:, :]
        del outputs
        torch.cuda.empty_cache()
        # Process through mapper
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
            a = self.mapper(gen_hidden)
            modulated_embeds = gen_hidden + a
        
        modulated_embeds.requires_grad_()

        # Generate with decoder2
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
            outputs2 = self.decoder2(
                input_ids=input_ids,
                context=modulated_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        return outputs2
    
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=2048, **kwargs):
        # Use decoder1.generate (with grad enabled) to collect hidden states
        if hasattr(self.decoder1.generate, '__wrapped__'):
            self.decoder1.generate = self.decoder1.generate.__wrapped__.__get__(self.decoder1, type(self.decoder1))
        with torch.enable_grad():
            outputs = self.decoder1.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        # Use only the new tokens' hidden states from the last layer
        gen_hidden = outputs.hidden_states[-1][:, input_ids.shape[1]:, :]  # [batch, new_tokens, hidden]
        a = self.mapper(gen_hidden)
        modulated_embeds = gen_hidden + a
        modulated_embeds.requires_grad_()

        # Generate final tokens using decoder2, conditioning on modulated embeddings
        if hasattr(self.decoder2.generate, '__wrapped__'):
            self.decoder2.generate = self.decoder2.generate.__wrapped__.__get__(self.decoder2, type(self.decoder2))
        with torch.enable_grad():
            gen2 = self.decoder2.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                context=modulated_embeds,
                **kwargs
            )
        return gen2

    def generate_with_no_grad(self, input_ids=None, attention_mask=None, max_new_tokens=2048, **kwargs):
        # Generate tokens from decoder1 (inference-only, no gradients)
        if hasattr(self.decoder1.generate, '__wrapped__'):
            self.decoder1.generate = self.decoder1.generate.__wrapped__.__get__(self.decoder1, type(self.decoder1))
        with torch.no_grad():
            outputs = self.decoder1.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        gen_hidden = outputs.hidden_states[-1][:, input_ids.shape[1]:, :]  # [batch, new_tokens, hidden]
        a = self.mapper(gen_hidden)
        modulated_embeds = gen_hidden + a

        # Generate tokens from decoder2, no gradients
        if hasattr(self.decoder2.generate, '__wrapped__'):
            self.decoder2.generate = self.decoder2.generate.__wrapped__.__get__(self.decoder2, type(self.decoder2))
        with torch.no_grad():
            gen2 = self.decoder2.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                context=modulated_embeds,
                **kwargs
            )
        return gen2

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # Save config once globally for the combined model
        self.config.save_pretrained(save_directory)

        # Save decoder1 and decoder2 separately
        self.decoder1.save_pretrained(os.path.join(save_directory, "decoder1"))
        self.decoder2.save_pretrained(os.path.join(save_directory, "decoder2"))

        # Save mapper
        torch.save(self.mapper.state_dict(), os.path.join(save_directory, "mapper.pt"))

        # Optional metadata
        with open(os.path.join(save_directory, "model_type.txt"), "w") as f:
            f.write("combined_model")

    @classmethod
    def from_pretrained(cls, load_directory, model_config=None, **kwargs):
        decoder1_dir = os.path.join(load_directory, "decoder1")
        decoder2_dir = os.path.join(load_directory, "decoder2")
        mapper_path = os.path.join(load_directory, "mapper.pt")

        # Load config from decoder1 (or global folder)
        if model_config is None:
            model_config = AutoConfig.from_pretrained(decoder1_dir, trust_remote_code=True)
        
        model_name = model_config._name_or_path
        hidden_dim = model_config.hidden_size

        # Load decoders with specified kwargs
        decoder1 = AutoModelForCausalLM.from_pretrained(
            decoder1_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # can also be set through config
            **kwargs
        )

        decoder2 = AutoModelForCausalLM.from_pretrained(
            decoder2_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            **kwargs
        )

        # Build model
        model = cls(model_config, model_name, decoder1, decoder2, hidden_dim)

        # Load mapper weights
        if os.path.exists(mapper_path):
            model.mapper.load_state_dict(torch.load(mapper_path, map_location="cpu"))
        else:
            print("[WARNING] mapper.pt not found. Mapper weights will be randomly initialized.")

        return model

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for the generate method, handling different generation strategies."""
        attention_mask = kwargs.get("attention_mask", None)
        
        # Initial step: get hidden states from decoder1
        if past_key_values is None:
            # First step - standard preparation
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "use_cache": kwargs.get("use_cache", False),
            }
        else:
            # Subsequent steps - use cached computation
            return {
                "input_ids": input_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
            }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) 
                                    for past_state in layer_past),)
        return reordered_past

    def enable_mapper_gradient_checkpointing(self):
        self.mapper_gradient_checkpointing = True

    def disable_mapper_gradient_checkpointing(self):
        self.mapper_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.decoder1, "gradient_checkpointing_enable"):
            self.decoder1.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.decoder2, "gradient_checkpointing_enable"):
            self.enable_mapper_gradient_checkpointing = True
            self.decoder2.gradient_checkpointing_enable(**kwargs)
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.decoder1, "gradient_checkpointing_disable"):
            self.decoder1.gradient_checkpointing_disable()
        if hasattr(self.decoder2, "gradient_checkpointing_disable"):
            self.enable_mapper_gradient_checkpointing = False
            self.decoder2.gradient_checkpointing_disable()
        self.config.gradient_checkpointing = False


def unwrap_model(model):
    return getattr(model, "module", model)

def save_combined_model_safely(
    model,
    tokenizer,
    save_dir="output_combined_model",
    trainer=None,
    save_trainer_state=True
):
    os.makedirs(save_dir, exist_ok=True)
    model = unwrap_model(model)
    model.decoder1.save_pretrained(os.path.join(save_dir, "decoder1"))
    model.decoder2.save_pretrained(os.path.join(save_dir, "decoder2"))
    torch.save(model.mapper.state_dict(), os.path.join(save_dir, "mapper.pt"))
    tokenizer.save_pretrained(save_dir)
    model.config.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "model_type.txt"), "w") as f:
        f.write("combined_model")
    print(f"[INFO] Combined model saved to: {save_dir}")

def load_combined_model(
    load_dir="output_combined_model",
    map_location="cpu",
    torch_dtype=torch.float16,
    trust_remote_code=True
):
    decoder1_dir = os.path.join(load_dir, "decoder1")
    decoder2_dir = os.path.join(load_dir, "decoder2")
    mapper_path = os.path.join(load_dir, "mapper.pt")
    config = AutoConfig.from_pretrained(decoder1_dir, trust_remote_code=trust_remote_code)
    model_name = config._name_or_path
    hidden_dim = config.hidden_size
    decoder1 = AutoModelForCausalLM.from_pretrained(
        decoder1_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )
    decoder2 = AutoModelForCausalLM.from_pretrained(
        decoder2_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )
    model = CombinedModel(config, model_name, decoder1, decoder2, hidden_dim)
    if os.path.exists(mapper_path):
        model.mapper.load_state_dict(torch.load(mapper_path, map_location=map_location))
    else:
        print(f"[WARNING] Mapper weights not found at: {mapper_path}")

    print(f"[INFO] Loaded CombinedModel from: {load_dir}")
    return model

def custom_collate_fn(batch):
    collated = {}
    for k in batch[0]:
        values = [item[k] for item in batch]
        # Convert lists to tensors if needed
        if isinstance(values[0], torch.Tensor):
            pass
        elif isinstance(values[0], list):
            values = [torch.tensor(v) for v in values]
        else:
            # For non-sequence fields (str, int, float, etc), keep as is or make a tensor if desired
            pass
        collated[k] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values

        if isinstance(collated[k], torch.Tensor) and collated[k].dim() == 3 and collated[k].shape[1] == 1:
            collated[k] = collated[k].squeeze(1)
    return collated

def Train_stage1(model, train_ds, eval_ds, tokenizer):
    from trl import SFTTrainer, SFTConfig

    train_args = SFTConfig(
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        torch_compile=False,
        deepspeed="./run_config/ds_config.json",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=4,
        save_steps=16,
        output_dir="./output_decoder1_test/",
        save_strategy="steps",
        eval_strategy="steps",
        save_safetensors=False,
        save_only_model=True,
        packing=True,
        run_name=f"test_stage1-{dt}",
        #report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        processing_class=tokenizer,
        #data_collator=ds_collate,
    )
    print("Trainer stage 1 initialized.. starting training")
    trainer.train()
    print("Training stage 1 completed. Saving model...")
    trainer.save_model("./output_decoder1_test/")
    print("Model stage 1 saved successfully.")

def Train_stage2(model, train_ds, eval_ds, tokenizer):
    from trl import SFTTrainer, SFTConfig

    train_args = SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        torch_compile=False,
        deepspeed="./run_config/ds_config.json",
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=100,
        output_dir="./output_combined_model_stage2_test/",
        save_strategy="steps",
        eval_strategy="steps",
        save_safetensors=False,
        save_only_model=True,
        packing=True,
        run_name=f"test_stage2-{dt}",
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        processing_class=tokenizer
    )
    print("Trainer stage 2 initialized.. starting training")
    trainer.train()
    print("Training stage 2 completed. Saving model...")
    trainer.save_model("./output_combined_model_stage2_test/")
    print("Model stage 2 saved successfully.")

def Train_stage3(model, train_ds, eval_ds, tokenizer):
    from trl import SFTTrainer, SFTConfig
    
    train_args = SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        torch_compile=False,
        deepspeed="./run_config/ds_config.json",
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=40,
        save_steps=100,
        output_dir="./output_combined_model_stage3/",
        save_strategy="steps",
        eval_strategy="steps",
        save_safetensors=False,
        save_only_model=True,
        packing=True,
        run_name=f"test_stage3-{dt}",
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        processing_class=tokenizer
    )
    print("Trainer stage 3 initialized.. starting training")
    trainer.train()
    print("Training stage 3 completed. Saving model...")
    trainer.save_model("./output_combined_model_stage3/")
    print("Model stage 3 saved successfully.")

def Train_stage4(model, train_ds, eval_ds, tokenizer):
    from trl import GRPOTrainer, GRPOConfig
    import numpy as np
    import re

    def compute_reward(completion, reference_kernel=None, example=None):
        reward = 0.0
        cuda_keywords = ["__global__", "__device__", "cudaMemcpy", "cudaMalloc", "blockIdx", "threadIdx"]

        # Reward for presence of <think> (low reward)
        if "<think>" in completion and "</think>" in completion:
            reward += 1.0

        # Reward for presence of <code_cuda> (medium reward)
        code_cuda_match = re.search(r"<code_cuda>(.*?)</code_cuda>", completion, re.DOTALL)
        if code_cuda_match:
            reward += 4.0
            code_cuda_content = code_cuda_match.group(1)

            # High reward if CUDA keywords are found within <code_cuda>...</code_cuda>
            for kw in cuda_keywords:
                if kw in code_cuda_content:
                    reward += 1.0
        else:
            code_cuda_content = ""

        # Low reward if CUDA keywords appear **outside** <code_cuda>...</code_cuda>
        for kw in cuda_keywords:
            # Find all occurrences outside <code_cuda>
            if kw in completion and kw not in code_cuda_content:
                reward += 0.5

        return reward

    # Define the reward function for GRPOTrainer
    def reward_fn(prompts, completions, samples=None, **kwargs):
        rewards = []
        for prompt, completion, sample in zip(prompts, completions, samples or [{}]*len(prompts)):
            cpp_code = sample.get("cpp_code", "")
            reference = sample.get("cuda_code", None)
            reward = compute_reward(completion, reference_kernel=reference, example=sample)
            rewards.append(reward)
        return rewards

    train_args = GRPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        num_train_epochs=1,
        torch_compile=False,
        #deepspeed="./deepspeed_config/ds_config_stage4.json",
        fp16=True,
        # gradient_checkpointing=True,
        logging_steps=40,
        save_steps=100,
        output_dir="./output_combined_model_GRPO/",
        save_strategy="steps",
        eval_strategy="steps",
        save_safetensors=False,
        save_only_model=True,
        run_name=f"test_stage4-{dt}",
        report_to="wandb",
    )
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        reward_funcs=reward_fn,
        processing_class=tokenizer
    )
    print("Trainer initialized.. starting training for GRPO")
    trainer.train()
    print("GRPO Training completed. Saving model...")
    trainer.save_model("./output_combined_model_GRPO/")
    print("Model saved successfully.")

