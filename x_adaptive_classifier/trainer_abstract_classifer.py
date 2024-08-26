from datetime import datetime
import argparse
import csv
import logging
import os

from datasets import Dataset
from evaluate import load as load_metric
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import chardet
import matplotlib.pyplot as plt
import pandas as pd
import torch


LOGGERS = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in LOGGERS:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


ACCURACY_METRIC = load_metric("accuracy", trust_remote_code=True)


def validate_data_table_path(path):
    try:
        with open(path, 'r') as file:
            header = file.readline().strip().split(',')
            if not all(column in header for column in ['include', 'abstract']):
                raise argparse.ArgumentTypeError(
                    f'"{path}" file must contain "include" and "abstract" columns.')
    except Exception as e:
        raise argparse.ArgumentTypeError(f'Failed to read CSV file: {e}')

    return path


class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'accuracy', 'loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = state.epoch
        accuracy = metrics.get('eval_accuracy')
        loss = metrics.get('eval_loss')

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, accuracy, loss])


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return ACCURACY_METRIC.compute(predictions=preds, references=p.label_ids)


def main():
    parser = argparse.ArgumentParser(description="Train a classifier for include/exclude on abstracts for CONVEI project.")
    parser.add_argument(
        'data_table_path',
        type=validate_data_table_path,
        help="Path to the CSV file containing 'include' and 'abstract' columns.")

    args = parser.parse_args()
    print(f'cuda is available: {torch.cuda.is_available()}')
    with open(args.data_table_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    print(f'encoding is: {encoding}')
    table = pd.read_csv(args.data_table_path, encoding=encoding)
    y = table['include']
    X = table['abstract']

    # Initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply undersampling to the training set
    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train.to_frame(), y_train)

    # Identify the samples that were tossed out
    tossed_samples_mask = ~X_train.index.isin(X_train_balanced.index)
    X_tossed = X_train[tossed_samples_mask]
    y_tossed = y_train[tossed_samples_mask]

    # Add tossed samples to the test set
    X_test_extended = pd.concat([X_test, X_tossed])
    y_test_extended = pd.concat([y_test, y_tossed])
    print(X_train_balanced)
    print(X_train)
    train_df = pd.DataFrame({
        'text': X_train_balanced.squeeze(),
        'label': y_train_balanced.squeeze(),
    })

    test_df = pd.DataFrame({
        'text': X_test_extended.squeeze(),
        'label': y_test_extended.squeeze(),
    })

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length', truncation=True,
            max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])
    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    num_epochs = 100
    batch_size = 16
    num_batches_per_epoch = len(train_dataset) // batch_size
    total_steps = num_epochs * num_batches_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps

    training_args = TrainingArguments(
        output_dir='./final_results',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir='./final_logs',
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CSVLoggerCallback(log_file='training_log.csv')],
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    trainer.train()

    saved_model_name = (
        f'convei_abstract_classifier_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', saved_model_name)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    eval_results = trainer.evaluate()
    print(eval_results)

    for name, dataset in [
            ('eval', tokenized_eval_dataset),
            ('train', tokenized_train_dataset)]:
        predictions = trainer.predict(dataset)
        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f'{name}.png')
        plt.close()
        plt.show()


if __name__ == '__main__':
    main()
