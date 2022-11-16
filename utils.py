from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import nn
from torch import optim

from lstm_glove import LSTM_GLove

import torch

def _custom_collate(data: list):

    # Get original input length for pack padded sequence
    input_len = [len(d['text']) for d in data]
    input_len = torch.tensor(input_len)

    # For padding process
    inputs = [torch.tensor(d['text']) for d in data]
    inputs_padded = pad_sequence(inputs, batch_first=True)

    # For labels
    labels = [d['class'] for d in data]
    labels = torch.tensor(labels)

    return input_len, inputs_padded, labels

def split_training_testing(x_full, y_full):

    # 60% for training, 20% for validation, 20% for testing (VAT)
    # 60% for training, 40% for testing (Non-VAT)
    x_train, x_test = random_split(
        x_full, 
        [0.6, 0.4],
        generator=torch.Generator().manual_seed(42)
    )

    y_train, y_test = random_split(
        y_full, 
        [0.6, 0.4],
        generator=torch.Generator().manual_seed(42)
    )

    return x_train, y_train, x_test, y_test

def transform_dataloader(dataloader_dataset):
    return DataLoader(dataloader_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=_custom_collate
    )

def multi_accuracy_calculation(prediction, actual):

    y_pred_softmax = torch.log_softmax(prediction, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == actual).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)

    return acc

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

    accuracy_stats = { 'train' : []}
    loss_stats = {'train': []}
    epochs = 5

    for epoch in range( (1, epochs + 1)):
        
        model.train()

        train_epoch_loss = 0
        train_epoch_acc = 0

        for (x_original_len, x_padded, y_train) in train_data:
            
            # Load inputs and labels onto device
            x_padded, y_train = x_padded.to(device), y_train.to(device)

            optimizer.zero_grad()

            # Predict the outputs
            y_pred = model(x_padded, x_original_len)

            # Compute the loss and accuracy
            train_loss = criterion(y_pred.squeeze(), y_train.float())
            train_acc = multi_accuracy_calculation(y_pred, y_train)

            # Back propagation
            # Update for parameters and compute the updates
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

            # Update the dictionary losses
            accuracy_stats['train'].append(train_epoch_acc/len(train_data))
            loss_stats['train'].append(train_epoch_loss/len(train_data))
        
    return model
