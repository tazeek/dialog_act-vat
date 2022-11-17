from torch import nn
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from torch import nn
from torch import optim
from tqdm.notebook import tqdm

from lstm_glove import LSTM_GLove

import torch

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

def train_model(train_data, glove_embeddings):

    # Prepare the model
    model = LSTM_GLove(glove_embeddings)
    print(model)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    # Use GPU, if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mount model onto the GPU
    model.to(device)

    epochs = 20
    train_set_size = len(train_data)

    for epoch in range(1, epochs + 1):
        
        model.train()

        train_epoch_loss = 0

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
        
        # Print every 10 epochs
        if epoch % 10 == 0:
            print(
                f'Epoch {epoch+0:03}: |'
                f' Train Loss: {train_epoch_loss/train_set_size:.5f} | '
                f' Train Acc: {train_epoch_acc/train_set_size:.3f} | '
                f' Train F1: {train_epoch_f1/train_set_size:.3f} | '
            )
        
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
            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

            y_test_list.append(y_test.cpu().numpy())
            y_test_list = [a.squeeze().tolist() for a in y_test_list]
    
    return None
