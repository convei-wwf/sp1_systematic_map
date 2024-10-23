from datetime import datetime
import argparse
import csv
import logging
import os

from transformers import XLNetConfig
from datasets import Dataset
from evaluate import load as load_metric
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import default_data_collator

import chardet
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader


LOGGERS = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in LOGGERS:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


CONFUSION_FIGS = 'confusion_figs'
os.makedirs(CONFUSION_FIGS, exist_ok=True)
ACCURACY_METRIC = load_metric("accuracy", trust_remote_code=True)
CLASS_WEIGHTS = [1.0, 5.0]  # Adjust weights as needed

class WeightedXLNetForSequenceClassification(XLNetForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get outputs from the parent class, excluding labels to avoid default loss computation
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Exclude labels to prevent default loss computation
            **kwargs
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                # Move class weights to the same device as logits
                class_weights = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            # Compute the loss with class weights
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return outputs using SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def validate_data_table_path(path):
    try:
        with open(path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        with open(path, 'r', encoding=encoding) as file:
            header = file.readline().strip().split(',')
            if not all(column in header for column in ['include', 'abstract']):
                raise argparse.ArgumentTypeError(
                    f'"{path}" file must contain "include" and "abstract" columns.')
    except Exception as e:
        raise argparse.ArgumentTypeError(f'Failed to read CSV file: {e}')

    return path


def log_gradient_norm(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} is frozen.")
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'Gradient Norm: {total_norm}')


class GradientNormCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        log_gradient_norm(model)

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file, steps_per_epoch, model_datetime):
        self.log_file = log_file
        self.model_datetime = model_datetime
        self.steps_per_epoch = steps_per_epoch
        self.accuracy_history = []
        self.loss_history = []
        self.epoch_history = []
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'accuracy', 'loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = state.global_step
        accuracy = metrics.get('eval_accuracy')
        loss = metrics.get('eval_loss')
        epoch = step / self.steps_per_epoch

        # self.accuracy_history.append(accuracy)
        # self.loss_history.append(loss)
        # self.epoch_history.append(epoch)

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, accuracy, loss])

        # # Plotting the graph
        # plt.figure()
        # plt.plot(self.epoch_history, self.accuracy_history, label='Accuracy')
        # plt.plot(self.epoch_history, self.loss_history, label='Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Metrics')
        # plt.legend()
        # plt.title(f'Training Metrics at Step {step}')
        # plt.savefig(f'metrics_plot_step_{step}_{self.model_datetime}.png')
        # plt.close()


# def compute_metrics(p):
#     preds = p.predictions.argmax(-1)
#     return ACCURACY_METRIC.compute(predictions=preds, references=p.label_ids)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train or continue training a classifier for include/exclude on abstracts for the CONVEI project.")
    parser.add_argument(
        'data_table_path',
        type=validate_data_table_path,
        help="Path to the CSV file containing 'include' and 'abstract' columns.")
    parser.add_argument(
        '--model_path',
        type=str,
        help="Optional path to a pre-trained model to continue training.")


    model_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
        X, y, test_size=0.2,
        #stratify=y,
        random_state=42)

    # Apply undersampling to the training set
    rus = RandomUnderSampler(
        sampling_strategy={0: sum(y_train == 1), 1: sum(y_train == 1)},
        random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(
        X_train.to_frame(), y_train)

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
    # model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

    config = XLNetConfig.from_pretrained('xlnet-base-cased', num_labels=2)
    if args.model_path is None:
        model = WeightedXLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            config=config,
            class_weights=CLASS_WEIGHTS
        )
    else:
        tokenizer = XLNetTokenizer.from_pretrained(args.model_path)
        config = XLNetConfig.from_pretrained(args.model_path)
        model = WeightedXLNetForSequenceClassification.from_pretrained(
            args.model_path,
            config=config,
            class_weights=CLASS_WEIGHTS
        )

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

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

    num_epochs = 25
    batch_size = 16

    steps_per_epoch = len(train_dataset) // batch_size
    gradient_accumulation_steps = steps_per_epoch

    total_steps = steps_per_epoch // gradient_accumulation_steps * num_epochs
    warmup_steps = int(0.1 * total_steps)

    class ConfusionMatrixCallback(TrainerCallback):
        def __init__(self, model_datetime, train_dataset, eval_dataset):
            self.model_datetime = model_datetime
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            step = state.global_step
            eval_accuracy = metrics.get('eval_accuracy')
            eval_loss = metrics.get('eval_loss')
            model = kwargs['model']
            device = model.device

            # Move model to evaluation mode
            model.eval()

            # Create DataLoaders
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=default_data_collator)
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=default_data_collator)

            # Function to compute predictions and confusion matrix
            def compute_confusion_matrix_and_accuracy(dataloader):
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in dataloader:
                        # Move inputs and labels to the device
                        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                        labels = batch['labels'].to(device)
                        outputs = model(**inputs)
                        logits = outputs.logits
                        all_preds.append(logits.detach().cpu())
                        all_labels.append(labels.cpu())

                predictions = torch.cat(all_preds)
                labels = torch.cat(all_labels)

                # Convert to numpy arrays
                y_pred = predictions.argmax(-1).numpy()
                y_true = labels.numpy()

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                # Compute accuracy
                acc = accuracy_score(y_true, y_pred)
                return cm, acc

            # Compute confusion matrices and accuracies
            cm_eval, acc_eval = compute_confusion_matrix_and_accuracy(eval_dataloader)
            cm_train, acc_train = compute_confusion_matrix_and_accuracy(train_dataloader)

            # Plotting confusion matrices side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            disp_eval = ConfusionMatrixDisplay(confusion_matrix=cm_eval)
            disp_eval.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
            axes[0].set_title(f'Eval Confusion Matrix at Step {step}\nAccuracy: {acc_eval:.4f}, Loss: {eval_loss:.4f}')

            disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
            disp_train.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
            axes[1].set_title(f'Train Confusion Matrix at Step {step}\nAccuracy: {acc_train:.4f}')

            plt.tight_layout()
            os.makedirs(f'{CONFUSION_FIGS}/{self.model_datetime}', exist_ok=True)
            plt.savefig(f'{CONFUSION_FIGS}/{self.model_datetime}/confusion_matrix_step_{step}_{self.model_datetime}.png')
            plt.close()

    training_args = TrainingArguments(
        output_dir=f'./model_steps/{model_datetime}',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-7,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir='./final_logs',
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        max_grad_norm=0.5,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            CSVLoggerCallback(
                log_file=f'training_log_{model_datetime}.csv',
                steps_per_epoch=1,
                model_datetime=model_datetime
            ),
            ConfusionMatrixCallback(
                model_datetime=model_datetime,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_eval_dataset
            )],
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    trainer.train()

    saved_model_name = (
        f'convei_abstract_classifier_{model_datetime}')
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', saved_model_name)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == '__main__':
    main()
