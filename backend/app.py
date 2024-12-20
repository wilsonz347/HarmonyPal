import math

from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

# Load pre-trained models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Preprocess the dataset
def preprocessing_data(ds):
    ds[['question', 'answer']] = ds['text'].str.split('<ASSISTANT>:', n=1, expand=True)

    ds['question'] = ds['question'].str.replace('<HUMAN>:', '').str.strip()
    ds['answer'] = ds['answer'].str.strip()

    ds.drop('text', axis=1, inplace=True)

    return ds

def fine_tune_response_generation_model(train_dataset, val_dataset):
    """Fine-tune GPT-2 for response generation"""
    training_args = TrainingArguments(
        output_dir='./response_generation_results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=gpt2_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=gpt2_tokenizer,
            mlm=False
        )
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model('./fine_tuned_response_model')
    return gpt2_model

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
    ds = pd.read_parquet("hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")

    ds = preprocessing_data(ds)

    # Create data split
    train_df, val_df = train_test_split(ds, test_size=0.2)

    # Create datasets
    train_dataset = MentalHealthDataset(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        gpt2_tokenizer
    )

    val_dataset = MentalHealthDataset(
        val_df['question'].tolist(),
        val_df['answer'].tolist(),
        gpt2_tokenizer
    )

    # Fine-tune the model
    trainer = fine_tune_response_generation_model(train_dataset, val_dataset)

    loss = trainer.evaluate()['eval_loss']
    perplexity = math.exp(loss)
    print(f"Perplexity: {perplexity}")

if __name__ == "__main__":
    main()