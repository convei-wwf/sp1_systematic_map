import csv
import logging

from transformers import get_linear_schedule_with_warmup
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from transformers import TrainerCallback
from datasets import Dataset
from evaluate import load as load_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, EarlyStoppingCallback
import chardet
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

DATA_TABLE_PATH = 'tmp_training_set.csv'


LOGGERS = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in LOGGERS:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


ACCURACY_METRIC = load_metric("accuracy", trust_remote_code=True)


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
    print(f'cuda is available: {torch.cuda.is_available()}')
    with open(DATA_TABLE_PATH, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    print(f'encoding is: {encoding}')
    table = pd.read_csv(DATA_TABLE_PATH, encoding=encoding)
    y = table['include']
    X = table['abstract']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    train_df = pd.DataFrame({
        'text': X_train,
        'label': y_train
    })

    test_df = pd.DataFrame({
        'text': X_test,
        'label': y_test
    })

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)

    model_name = "microsoft/deberta-large"  # Or "microsoft/deberta-large", "microsoft/deberta-v2-xlarge"
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
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

    class_weights = torch.tensor([
                1.0,
                len(table)/sum(table['include'] == True)])
    print(f'class weights: {class_weights}')

    class WeightedClassification(DebertaForSequenceClassification):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # Class weights favor proportion of classifications
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(logits.device))
            return (loss, outputs) if return_outputs else loss

    # Define final training arguments using the best hyperparameters

    num_epochs = 25
    batch_size = 16
    num_batches_per_epoch = len(train_dataset) // batch_size
    total_steps = num_epochs * num_batches_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps

    training_args = TrainingArguments(
        output_dir='./final_results',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-7,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir='./final_logs',
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",  # Use a linear learning rate scheduler with warmup
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=WeightedClassification.from_pretrained(model_name, num_labels=2),
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CSVLoggerCallback(log_file='training_log.csv')],
    )

    trainer.train()

    trainer.save_model("./my_model")
    tokenizer.save_pretrained("./my_model")

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
