from sklearn import metrics
import pandas as pd
import os
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils import EmotionDataset
from transformers import BertForSequenceClassification, BertTokenizer
import argparse


def load_dataset(filepath: str, text_column: str = 'Tweet', separator: str = '\t') -> EmotionDataset:
    dataset = pd.read_csv(filepath, sep=separator, index_col=0)
    label_names = list(dataset.columns[1:])

    data = dataset[text_column].values
    labels = dataset[label_names].values

    return EmotionDataset(data, labels, label_names)


def train(model, tokenizer, train_dl, valid_dl, epochs, lr, device):
    best_model = None
    best_loss = float("inf")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        for data, labels in tqdm(train_dl):
            inputs = tokenizer(data, padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(dtype=torch.float64).to(device)

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

            train_loss += loss.item() * len(data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_dl.dataset)
        train_losses.append(train_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data, labels in tqdm(valid_dl):
                inputs = tokenizer(data, padding=True, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(dtype=torch.float64).to(device)

                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)

                valid_loss += loss.item() * len(data)

        valid_loss /= len(valid_dl.dataset)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model

        print(
            f"epoch: {epoch:03d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")

    return best_model, train_losses, valid_losses


def test(model, test_dl, tokenizer, threshold, device):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for data, _ in tqdm(test_dl):
            inputs = tokenizer(data, padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.sigmoid(logits)
            probabilities += probs

    probabilities = torch.stack(probabilities)
    probabilities = probabilities.cpu()
    predictions = (probabilities > threshold).float().tolist()
    probabilities = probabilities.tolist()

    targets = test_dl.dataset.labels

    jaccard = metrics.jaccard_score(targets, predictions, average="samples")
    clf_report = metrics.classification_report(
        targets, predictions, target_names=test_dl.dataset.label_names, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(clf_report)
    report_df = report_df.T
    report_df['jaccard'] = jaccard

    predictions_df = pd.DataFrame(
        predictions, columns=test_dl.dataset.label_names)

    probabilities_df = pd.DataFrame(
        probabilities, columns=test_dl.dataset.label_names)

    return report_df, predictions_df, probabilities_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", default="../data/semeval/train.txt")
    parser.add_argument("--valid-data", default="../data/semeval/valid.txt")
    parser.add_argument("--test-data", default="../data/semeval/test.txt")
    parser.add_argument("--text-column", default="Tweet")
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--valid-batch-size", type=int, default=32)

    parser.add_argument("--pretrained-path",
                        default="../bert-models/bert-base-uncased-top-50-emoji")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output-dir", default="trained_bert_model")

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")

    # Load dataset
    train_dataset = load_dataset(
        args.train_data, args.text_column, args.sep)
    valid_dataset = load_dataset(
        args.valid_data, args.text_column, args.sep)
    test_dataset = load_dataset(args.test_data, args.text_column, args.sep)

    train_dl = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, args.valid_batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, args.valid_batch_size, shuffle=False)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_path, output_attentions=True, num_labels=len(train_dataset.label_names))
    model = model.to(device)

    model, train_losses, valid_losses = train(
        model, tokenizer, train_dl, valid_dl, args.epochs, 1e-5, device)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    losses_df = pd.DataFrame({"train": train_losses, "valid": valid_losses})
    losses_df.to_csv(os.path.join(args.output_dir, "losses.csv"))

    clf_report, preds, probs = test(model, test_dl, tokenizer, 0.5, device)

    clf_report.to_csv(os.path.join(args.output_dir, "metrics.csv"))
    preds.to_csv(os.path.join(args.output_dir, "predictions.csv"))
    probs.to_csv(os.path.join(args.output_dir, "probabilities.csv"))
