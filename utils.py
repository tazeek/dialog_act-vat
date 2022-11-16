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
    
    for epoch in range(4):
        
        for (original_lengths, padded_inputs, labels) in train_data:
            
            # Load inputs and labels
            # inputs, labels = inputs.to(device), labels.to(device)

            # Predict the outputs
            # y_pred, h = model(padded_inputs, original_lengths)

            # Compute the loss
            #loss = criterion(y_pred.squeeze(), labels.float())

            # Update for parameter
            #loss.backward()

            # Compute updates for parameters
            #optimizer.step()
            #optimizer.zero_grad()
            ...

    return None
