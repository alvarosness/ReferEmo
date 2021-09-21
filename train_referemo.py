import os
import json
from sklearn import metrics
import torch
from torch import nn, optim
from tqdm import tqdm
from models.referemo import ReferEmo
from torch.utils.data import DataLoader
from models.vocab import GloVe
from models.tokenizers import BERTTokenizer, BiLSTMTokenizer
from utils import DataProcessor, EmotionDataset
import pandas as pd
import argparse


def load_dataset(filepath: str, text_column: str = 'Tweet', separator: str = '\t') -> EmotionDataset:
    dataset = pd.read_csv(filepath, sep=separator, index_col=0)
    label_names = list(dataset.columns[1:])

    data = dataset[text_column].values
    labels = dataset[label_names].values

    return EmotionDataset(data, labels, label_names)


def train(model: nn.Module, train_dl: DataLoader, valid_dl: DataLoader, preprocessor, epochs: int, optimizer_name: str, lr: float, device: torch.device) -> nn.Module:
    best_model = None
    best_loss = float("inf")
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data, labels in tqdm(train_dl):
            # HACK
            # Need to convert this tuple of strings into a list of strings
            # so that the processor function can modify it
            data = list(data)

            model_input, labels = preprocessor.generate_batch_input(
                data, labels)
            model_input = model_input.to(device)
            labels = labels.to(device)

            logits, attns = model(model_input)
            loss = criterion(logits, labels)

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
                data = list(data)

                model_input, labels = preprocessor.generate_batch_input(
                    data, labels)
                model_input = model_input.to(device)
                labels = labels.to(device)

                logits, attns = model(model_input)
                loss = criterion(logits, labels)

                valid_loss += loss.item() * len(data)

        valid_loss /= len(valid_dl.dataset)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model

        print(
            f"epoch: {epoch:03d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")

    return best_model, train_losses, valid_losses


def test(model, test_dl, preprocessor, threshold, device):
    model.eval()
    probabilities = []
    attentions = []

    with torch.no_grad():
        for data, labels in tqdm(test_dl):
            data = list(data)

            model_input, _ = preprocessor.generate_batch_input(data, labels)
            model_input = model_input.to(device)

            logits, attns = model(model_input)
            probs = torch.sigmoid(logits)
            probabilities += probs
            attentions += attns

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

    return report_df, predictions_df, probabilities_df, attentions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="../data/semeval/train.txt")
    parser.add_argument("--valid-data", default="../data/semeval/valid.txt")
    parser.add_argument("--test-data", default="../data/semeval/test.txt")
    parser.add_argument("--text-column", default="Tweet")
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--model-config", default="config.json")
    parser.add_argument("--pretrained-bert", default="bert-base-uncased")
    parser.add_argument("--glove", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output-dir", default="trained_model")

    args = parser.parse_args()

    # Load dataset
    train_dataset = load_dataset(
        args.train_data, args.text_column, args.sep)
    valid_dataset = load_dataset(
        args.valid_data, args.text_column, args.sep)
    test_dataset = load_dataset(args.test_data, args.text_column, args.sep)

    # TODO
    # I could specify the collate_fn as the DataProcessor
    # or I could explicitly call the function at each iteration
    train_dl = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, args.valid_batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, args.valid_batch_size, shuffle=False)

    # Load tokenizer and all those things
    wv = GloVe(filepath=args.glove)
    text_tokenizer = BiLSTMTokenizer(wv.vocab)
    ref_tokenizer = BERTTokenizer()
    preprocessor = DataProcessor(text_tokenizer, ref_tokenizer)

    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load hyperparameters
    config = json.load(open(args.model_config))
    hdim = config['hdim']
    n_layers = config['n_layers']
    freeze_embeddings = config['freeze_embeddings']
    encoder_dropout_p = config['encoder_dropout_p']
    classification_dropout_p = config['clf_dropout_p']

    # Load model
    model = ReferEmo(
        embeddings=wv.get_embeddings(),
        nclasses=len(train_dataset.label_names),
        pretrained_bert=args.pretrained_bert,
        hdim=hdim,
        n_layers=n_layers,
        freeze_embeddings=freeze_embeddings,
        encoder_dropout_p=encoder_dropout_p,
        classification_dropout_p=classification_dropout_p
    )

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    model, train_losses, valid_losses = train(model, train_dl, valid_dl, preprocessor,
                                              args.epochs, config['optimizer'], config['lr'], device)

    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module, os.path.join(args.output_dir, "model.pt"))
    else:
        torch.save(model, os.path.join(args.output_dir, "model.pt"))

    clf_report, preds, probs, attns = test(
        model, test_dl, preprocessor, config['threshold'], device)

    losses_df = pd.DataFrame({"train": train_losses, "valid": valid_losses})
    losses_df.to_csv(os.path.join(args.output_dir, "losses.csv"))
    clf_report.to_csv(os.path.join(args.output_dir, "metrics.csv"))
    preds.to_csv(os.path.join(args.output_dir, "predictions.csv"))
    probs.to_csv(os.path.join(args.output_dir, "probabilities.csv"))

    for idx, attn in enumerate(attns):
        torch.save(attn, os.path.join(
            args.output_dir, f"attentions_{idx}.pth"))
