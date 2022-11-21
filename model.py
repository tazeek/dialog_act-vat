from torch import nn
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from torch import nn
from torch import optim

from sklearn.metrics import classification_report

from lstm_glove import LSTM_GLove
from vat_loss import VATLoss

import numpy as np
import pandas as pd

import torch

def _create_model(glove_embeddings):

    model = LSTM_GLove(glove_embeddings)

    return model

def _get_device():

    # Use GPU, if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return device

def _get_hyperparams():

    params = {
        'lr': 0.001,
        'epochs': 10,
        'alpha_val': 0.01
    }

    return params

def _vat_loss_calculation(model, device, validation_data):

    # For VAT Loss
    x_original_len_val, x_padded_val, _ = next(iter(validation_data))
    x_padded_val = x_padded_val.to(device)

    vat_loss = VATLoss()
    lds = vat_loss(model, x_padded_val, x_original_len_val)

    return lds

def _get_results_dictionary() -> dict:

    return {
        'epoch': [],
        'cross_entropy_loss': [],
        'accuracy': [],
        'f1': []
    }

def _save_csv_file(base_file, data) -> None:

    # Convert to dataframe
    df = pd.DataFrame(data)

    # Save dataframe to CSV
    base_file += '_training_results.csv'
    df.to_csv(base_file, index=False)

    return None

def metrics_evaluation(y_pred, y_train, device):

    # Get the predicted labels
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    accuracy = Accuracy().to(device)
    f1 = MulticlassF1Score(num_classes = 4).to(device)

    acc_score = accuracy(y_pred_tags, y_train)
    acc_score = torch.round(acc_score * 100)

    f1_score = f1(y_pred_tags, y_train)
    f1_score = torch.round(f1_score * 100)

    return acc_score, f1_score

def train_model(train_data, validation_data, glove_embeddings, base_filename):

    # Prepare the model
    model = _create_model(glove_embeddings)
    print(model)

    # Use GPU, if available
    device = _get_device()

    # Mount model onto the GPU
    model.to(device)

    # Get the hyperparameters
    hyper_params = _get_hyperparams()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr= hyper_params['lr'])

    epochs = hyper_params['epochs']
    alpha_val = hyper_params['alpha_val']
    train_set_size = len(train_data)

    # For storing results
    results_dict = _get_results_dictionary()

    for epoch in range(1, epochs + 1):
        
        model.train()

        train_epoch_loss = 0
        train_vat_loss = 0
        train_epoch_f1 = 0
        train_epoch_acc = 0

        for (x_original_len, x_padded, y_train) in train_data:
            
            # Convert to LongTensor
            y_train = y_train.type(torch.LongTensor)

            # Load inputs and labels onto device
            x_padded, y_train = x_padded.to(device), y_train.to(device)

            optimizer.zero_grad()

            # Predict the outputs
            y_pred = model(x_padded, x_original_len)

            # For VAT
            lds = _vat_loss_calculation(model, device, validation_data)

            # Compute the loss and accuracy
            train_loss = criterion(y_pred.squeeze(), y_train)
            train_acc, train_f1 = metrics_evaluation(y_pred, y_train, device)

            # Back propagation
            # Update for parameters and compute the updates
            train_loss.backward()
            optimizer.step()
            
            # Update for Display
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            train_epoch_f1 += train_f1.item()
            #train_vat_loss += lds.item()

        # Store the results
        results_dict['epoch'].append(epoch)
        results_dict['cross_entropy_loss'].append(train_epoch_loss)
        results_dict['accuracy'].append(train_epoch_acc)
        results_dict['f1'].append(train_epoch_f1)
        
        # Print every 10 epochs
        print(
            f'Epoch {epoch+0:03}: |'
            f' Train Loss: {train_epoch_loss/train_set_size:.5f} | '
            #f' VAT Loss: {(train_vat_loss * alpha_val)/train_set_size:.5f} | '
            f' Train Acc: {train_epoch_acc/train_set_size:.3f} | '
            f' Train F1: {train_epoch_f1/train_set_size:.3f} | '
        )

    # Save the CSV file
    _save_csv_file(results_dict, base_filename)

    return model


def test_model(test_data, model):

    # Use GPU, if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mount model onto the GPU
    model.to(device)

    y_pred_list = []
    y_test_list = []
    
    with torch.no_grad():
        
        model.eval()

        for (x_original_len, x_padded, y_test) in test_data:

            # Convert to LongTensor
            y_test = y_test.type(torch.LongTensor)

            # Load inputs and labels onto device
            x_padded, y_test = x_padded.to(device), y_test.to(device)

            # Predict the outputs
            y_pred = model(x_padded, x_original_len)

            # Get the predicted labels
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_test_list.append(y_test.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_test_list = [a.squeeze().tolist() for a in y_test_list]
    
    # Flatten list
    y_pred_list = [output for list in y_pred_list for output in list]
    y_test_list = [output for list in y_test_list for output in list]

    print(classification_report(y_test_list, y_pred_list))

    return None
