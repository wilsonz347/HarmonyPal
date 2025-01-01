import torch
import math
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, get_scheduler,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset


def set_up_model_tokenizer():
    # Load pre-trained models and tokenizers
    model_name = "gpt2"
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

    # Configure special tokens
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.padding_side = 'right'
    gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id

    gpt2_model.config.use_cache = False

    return gpt2_model, gpt2_tokenizer


# Preprocess the dataset
def preprocessing_data(ds):
    try:
        ds[['question', 'answer']] = ds['text'].str.split('<ASSISTANT>:', n=1, expand=True)

        ds['question'] = ds['question'].str.replace('<HUMAN>:', '').str.strip()
        ds['answer'] = ds['answer'].str.strip()

        ds.drop('text', axis=1, inplace=True)
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return None
    return ds


def r_training_args(batch_size=8, epochs=5, warmup_ratio=0.10):
    return TrainingArguments(
        output_dir='./response_generation_results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        logging_dir='./logs',
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        prediction_loss_only=True
    )


def r_trainer(model, tokenizer, train_dataset, val_dataset, training_args):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print("Model device:", next(model.parameters()).device)

        # Calculate total steps for training
        total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        warmup_steps = int(total_steps * training_args.warmup_ratio)

        optimizer = AdamW(
            model.parameters(),
            lr=3e-5,
            betas=(0.90, 0.999),
            eps=1e-8,
            weight_decay=0.05
        )

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers = (optimizer, scheduler),
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )

        # Train and save the model
        print("Starting training...")
        trainer.train()

        print("Training completed. Saving model...")
        trainer.save_model('./fine_tuned_mental_health_model')

        # Evaluate and return metrics
        print("Running evaluation...")
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])

        return trainer, eval_results, perplexity

    except Exception as e:
        print(f"Error details: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")


class MentalHealthDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.encodings = []
        for q, a in zip(questions, answers):
            text = f"Question: {q} Answer: {a}"
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'labels': self.encodings[idx]['input_ids']
        }

    def __len__(self):
        return len(self.encodings)


def main():
    try:
        ds = pd.read_parquet(
            "hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
        return None

    ds = preprocessing_data(ds)

    model, tokenizer = set_up_model_tokenizer()

    # Create data split
    train_df, val_df = train_test_split(ds, test_size=0.2)

    # Create datasets
    train_dataset = MentalHealthDataset(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        tokenizer
    )

    val_dataset = MentalHealthDataset(
        val_df['question'].tolist(),
        val_df['answer'].tolist(),
        tokenizer
    )

    # Fine-tune the model
    training_args = r_training_args()
    trainer, eval_results, perplexity = r_trainer(
        model, tokenizer, train_dataset, val_dataset, training_args
    )

    print(f"Training completed. Perplexity: {perplexity}")
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()