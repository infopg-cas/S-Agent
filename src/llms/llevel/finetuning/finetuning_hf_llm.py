import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

class LanguageModelTrainer:
    def __init__(self, model_path, dataset_path, output_dir):
        # init
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def load_pretrained_model(self, max_seq_length=2048, dtype=None, load_in_4bit=True):
        # Load the pre-trained model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=max_seq_length, # Set the maximum sequence length to 2048 tokens.
            dtype=dtype, # Set the data type to None for automatic detection.
            load_in_4bit=load_in_4bit, # Set the load_in_4bit flag to True to load the model weights in 4-bit precision.
        )

    # Create a PEFT model with the given parameters
    def configure_peft_model(self, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                             lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
                             random_state=3407, use_rslora=False, loftq_config=None):
        # Configure PEFT model
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

    def format_prompts(self, examples):
        # Format prompts
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["response"]
        texts = []
        alpaca_prompt = """
        ### Instruction:
        {}
        ### Input:
        {}
        ### Response:
        {}"""
        EOS_TOKEN = self.tokenizer.eos_token
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def train_model(self, max_seq_length, per_device_train_batch_size=2, gradient_accumulation_steps=4,
                    warmup_steps=5, max_steps=500, learning_rate=2e-4, logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
                    lr_scheduler_type="linear", seed=1337):
        # Load dataset
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        dataset = dataset.map(self.format_prompts, batched=True)

        # Train the model
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=logging_steps,
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                output_dir=self.output_dir,
            ),
        )
        # Train the model
        trainer.train()

    def inference(self, alpaca_prompt, inputs):
        # Perform inference
        FastLanguageModel.for_inference(self.model)
        text_streamer = TextStreamer(self.tokenizer)
        outputs = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
        return outputs

    def save_model(self, path):
        # Save the fine-tuned model
        self.model.save_pretrained(path)

