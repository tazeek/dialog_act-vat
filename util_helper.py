from models import bert_finetune
from tqdm import tqdm

import time
import logging
import argparse
import torch
import tomli

from sklearn.metrics import f1_score, recall_score, precision_score

def load_config_file():

    config_dict = {}

    with open("config.toml", mode="rb") as file:
        config_dict = tomli.load(file)

    return config_dict

def get_logger(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] " + \
        "[%(filename)s] " + \
        "[line:%(lineno)d] " + \
        "[%(levelname)s] %(message)s"
    )

    fh = logging.FileHandler(f'log_output/{log_file}.log', 'w')
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def create_parser():

    parser = argparse.ArgumentParser(description='Parser for VAT and Dialog Act')
    
    parser.add_argument('--model', default='lstm', metavar='model',
        help='Name of the model for training')

    parser.add_argument('--embed', default='glove', metavar='embed',
        help='Embeddings required for the model')
    
    parser.add_argument('--vat', default=None, metavar='vat',
        help='Check if the model requires VAT for Semi-supervision')

    return parser.parse_args()

def get_base_filename(args):

    # File name format: <model>_<embed>_<vat>
    base_filename = args.model

    if args.vat:
        base_filename += '_vat'
    
    return base_filename

def load_transformed_datasets(args, file):

    file_name = f'preprocessed_data/{file}_{args.embed}.pth'
    return torch.load(file_name)

def prepare_model_attributes(model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    return criterion, optimizer

def train_model(train_set, test_set):

    # Prepare device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(f'Using Device: {torch.cuda.get_device_name()}')
    else:
        print('Using CPU')

    # Get the model and put it on CUDA
    model = bert_finetune.BERT_FineTune(768, 4)
    model.to(device)

    trainable_params, nontrainable_params = 0, 0

    # Count the parameters
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            trainable_params += n_params
        else:
            nontrainable_params += n_params

    print(f"Trainable parameters: {trainable_params}\n, Non-trainable parameters: {nontrainable_params}")

    # Load the loss functions
    criterion, optimizer = prepare_model_attributes(model)

    start_time = time.time()
    epoch = 5

    for epoch in range(0, epoch + 1):

        print(f"Training Epoch: {epoch} \n")
        model.train()

        total_loss = 0

        epoch_start_time = time.time()

        for batch_data in tqdm(train_set, ncols=50):

            features = batch_data['features']
            labels = batch_data['labels']

            # Move to CUDA
            input_ids = features['input_ids'].to(device)
            mask = features['attention_mask'].to(device)
            labels = labels.to(device)

            # Get the output and losses
            output = model(input_ids, mask)
            loss = criterion(output, labels)
            total_loss += loss.cpu().item()
            
            # Update optimizer and loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Total loss: {total_loss}')
        print('\n\n')
        epoch_end_time = time.time() - epoch_start_time

        print(f"Epoch {epoch} training time: {epoch_end_time:.4f}")

        test_model(model, test_set, device)
        exit()
    
    end_time = time.time() - start_time
    print(f"Total Training Time: f{end_time:.4f} s")

    return None

def test_model(model, test_set, device):

    # Set in eval mode
    model.eval()

    actual, predicted = [], []

    for batch_data in tqdm(test_set, ncols=50):

        features = batch_data['features']
        labels = batch_data['labels']

        # Move to CUDA
        input_ids = features['input_ids'].to(device)
        mask = features['attention_mask'].to(device)

        actual.extend(labels)

        with torch.no_grad():
            pred_results = model(input_ids, mask)
            
        predicted.extend(pred_results.cpu())

    f_score = f1_score(actual, predicted, average="macro")
    r_score = recall_score(actual, predicted, average="macro")
    p_score = precision_score(actual, predicted, average="macro")

    print(f'F1-score: {f_score}\nRecall score: {r_score}\nPrecision score: {p_score}')

    return None
    ...