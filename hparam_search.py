import argparse
import json
import os
from functools import partial

import optuna
import pandas as pd
import torch
from optuna import visualization as viz
from optuna.trial import TrialState
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

from models.referemo import ReferEmo
from models.tokenizers import BERTTokenizer, BiLSTMTokenizer
from models.vocab import GloVe
from utils import DataProcessor, EmotionDataset


def partial_objective(embeddings, pretrained_bert, train_data, valid_data, preprocessor, epochs, device, trial):
    # Create model
    hdim = trial.suggest_categorical("hdim", [128, 256, 512, 1024, 2096])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    freeze_embeddings = trial.suggest_categorical(
        "freeze_embeddings", [True, False])
    encoder_dropout_p = trial.suggest_float("encoder_dropout_p", 0.1, 0.7)
    clf_dropout_p = trial.suggest_float("clf_dropout_p", 0.1, 0.7)

    model = ReferEmo(
        embeddings,
        len(train_data.label_names),
        pretrained_bert,
        hdim,
        n_layers,
        freeze_embeddings,
        encoder_dropout_p,
        clf_dropout_p
    )
    model.to(device)

    # Create optimizer
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Create dataset loaders
    train_dl = DataLoader(train_data, 32, shuffle=True)
    valid_dl = DataLoader(valid_data, 32)
    targets = valid_data.labels

    criterion = nn.BCEWithLogitsLoss()

    threshold = trial.suggest_float("threshold", 0.3, 0.5, step=0.1)

    # Train the model
    for epoch in range(epochs):
        model.train()

        for data, labels in train_dl:
            optimizer.zero_grad()
            data = list(data)

            model_input, labels = preprocessor.generate_batch_input(
                data, labels)
            model_input = model_input.to(device)
            labels = labels.to(device)
            logits, _ = model(model_input)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        probabilities = []

        with torch.no_grad():
            for data, labels in valid_dl:
                data = list(data)

                model_input, labels = preprocessor.generate_batch_input(
                    data, labels)
                model_input = model_input.to(device)
                labels = labels.to(device)

                logits, attns = model(model_input)
                probs = torch.sigmoid(logits)
                probabilities += probs.cpu()

        probabilities = torch.stack(probabilities)
        predictions = (probabilities >= threshold).float().tolist()
        f1_score = metrics.f1_score(targets, predictions, average="macro")

        trial.report(f1_score, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return f1_score


def load_dataset(filepath: str, text_column: str = 'Tweet', separator: str = '\t') -> EmotionDataset:
    dataset = pd.read_csv(filepath, sep=separator, index_col=0)
    label_names = list(dataset.columns[1:])

    data = dataset[text_column].values
    labels = dataset[label_names].values

    return EmotionDataset(data, labels, label_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--pretrained-bert", default="bert-base-uncased")
    parser.add_argument("--train-data", default="../data/semeval/train.txt")
    parser.add_argument("--valid-data", default="../data/semeval/valid.txt")
    parser.add_argument("--glove", default=None)
    parser.add_argument("--study-name", default="hparam_search")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output-dir", default="hparam_search_results")

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")

    train_data = load_dataset(args.train_data)
    valid_data = load_dataset(args.valid_data)

    wv = GloVe(filepath=args.glove)
    text_tokenizer = BiLSTMTokenizer(wv.vocab)
    ref_tokenizer = BERTTokenizer()
    preprocessor = DataProcessor(text_tokenizer, ref_tokenizer)

    objective = partial(
        partial_objective,
        wv.get_embeddings(),
        args.pretrained_bert,
        train_data,
        valid_data,
        preprocessor,
        args.epochs,
        device
    )

    study = optuna.create_study(
        study_name=args.study_name, direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    completed_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(completed_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    os.makedirs(args.output_dir)

    json.dump(trial.params, open(os.path.join(
        args.output_dir, "params.json"), "w"))

    viz.plot_param_importances(study).write_image(
        os.path.join(args.output_dir, "hparam_importance.png"))
    viz.plot_contour(study).write_image(os.path.join(
        args.output_dir, "contour.png"), width=1920, height=1080)
    viz.plot_parallel_coordinate(study).write_html(
        os.path.join(args.output_dir, "parallel_coords.html"))
    viz.plot_intermediate_values(study).write_html(
        os.path.join(args.output_dir, "learning_curve.html"))
    viz.plot_optimization_history(study).write_image(
        os.path.join(args.ouput_dir, "optimization_history.png"))
