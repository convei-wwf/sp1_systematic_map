import argparse
import logging
import os

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import XLNetForSequenceClassification, XLNetTokenizer
import chardet
import pandas as pd
import torch

# transformer loggers are noisy, this quiets them
LOGGERS = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in LOGGERS:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


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


def main():
    parser = argparse.ArgumentParser(
        description="Apply an include/exclude classifier to a table of abstracts.")
    parser.add_argument(
        'model_path',
        type=str,
        help="Path to the NLP model.")
    parser.add_argument(
        'data_table_path',
        type=validate_data_table_path,
        help="Path to the CSV file containing 'abstract' column.")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    print(f'cuda is available: {cuda_available}')

    model = XLNetForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    tokenizer = XLNetTokenizer .from_pretrained(args.model_path)

    print(f'{args.model_path} loaded')
    with open(args.data_table_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    print(f'encoding is: {encoding}')
    table = pd.read_csv(args.data_table_path, encoding=encoding)

    # Tokenize the 'abstract' column
    inputs = tokenizer(
        table['abstract'].tolist(), padding=True, truncation=True, return_tensors="pt")
    print('abstracts are tokenized')

    if cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'moving model to {device}')

    model.to(device)

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset),
        batch_size=args.batch_size)
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device)}
            outputs = model(
                input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device))
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(
                predictions.cpu().numpy())

    table['classification_prediction'] = all_predictions
    target_table_path = f'predicted_{os.path.basename(args.data_table_path)}'
    table.to_csv(target_table_path, index=False)


if __name__ == '__main__':
    main()
