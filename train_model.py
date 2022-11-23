from models import lstm_glove
from vat_loss import VATLoss
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from torch import nn
from torch import optim

import pandas as pd
import torch

class Model():

    def __init__(self, params):

        args = params['args']

        # Hyperparams
        self._lr = 0.001
        self._epochs = 10
        self._alpha_val = 0.01

        # Data Generators
        self._train_data = params['training']
        self._test_data = params['test']
        self._unlabeled_data = params['valid']

        # Others
        self._file_name = None

        # Results evaluation
        self._eval_results = self._get_results_dictionary()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model and mount to device
        self._model = self._create_model()
        self._model.to(self._device)
    
    def metrics_evaluation(y_pred, y_train, device):

        # TODO: Check if metrics are calculated properly
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

    def _save_csv_file(self, data, base_file) -> None:

        # Convert to dataframe
        df = pd.DataFrame(self._eval_results)

        # Save dataframe to CSV
        file_name = 'results/' + self._eval_results + '_training_results.csv'

        df.to_csv(file_name, index=False)

        return None

    def _get_results_dictionary(self):

        return {
            'epoch': [],
            'cross_entropy_loss': [],
            'accuracy': [],
            'f1': []
        }

    def _create_model(self):

        model = lstm_glove.LSTM_GLove()

        return model

    def _vat_loss_calculation(self, model, device, validation_data):

        # Create the VAT formula and test:
        # - DailyDialog's validation set
        # - Unlabeled data
        # - Both
        # - Check if VAT loss is actually correct or not (Refer to Paper)

        # For VAT Loss
        x_original_len_val, x_padded_val, _ = next(iter(validation_data))
        x_padded_val = x_padded_val.to(device)

        vat_loss = VATLoss()
        lds = vat_loss(model, x_padded_val, x_original_len_val)

        return lds

    def start_train(self, train_data):

        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Define Optimizer
        optimizer = optim.Adam(
            self._model.parameters(), 
            lr= self._lr
        )

        train_set_size = len(self._train_data)

        for epoch in range(1, self._epochs + 1):

            self._model.train()

            train_epoch_loss = 0
            train_vat_loss = 0
            train_epoch_f1 = 0
            train_epoch_acc = 0

            for (x_original_len, x_padded, y_train) in self._train_data:

                # Convert to LongTensor
                y_train = y_train.type(torch.LongTensor)

                # Load inputs and labels onto device
                x_padded, y_train = x_padded.to(self._device), y_train.to(self._device)

                optimizer.zero_grad()

                # Predict the outputs
                y_pred = self._model(x_padded, x_original_len)

                # For VAT
                #lds = _vat_loss_calculation(model, device, validation_data)

                # Compute the loss and accuracy
                train_loss = criterion(y_pred.squeeze(), y_train)
                train_acc, train_f1 = self._metrics_evaluation(y_pred, y_train, self._device)

                # Back propagation
                # Update for parameters and compute the updates
                train_loss.backward()
                optimizer.step()
                
                # Update for Display
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
                train_epoch_f1 += train_f1.item()
                #train_vat_loss += lds.item()
            
            # Normalize results
            train_epoch_loss = train_epoch_loss/train_set_size
            train_epoch_acc = train_epoch_acc/train_set_size
            train_epoch_f1 = train_epoch_f1/train_set_size

            # Store the results
            self._eval_results['epoch'].append(epoch)
            self._eval_results['cross_entropy_loss'].append(train_epoch_loss)
            self._eval_results['accuracy'].append(train_epoch_acc)
            self._eval_results['f1'].append(train_epoch_f1)
            
            # Print every 10 epochs
            print(
                f'Epoch {epoch+0:03}: |'
                f' Train Loss: {train_epoch_loss:.5f} | '
                #f' VAT Loss: {(train_vat_loss * alpha_val)/train_set_size:.5f} | '
                f' Train Acc: {train_epoch_acc:.3f} | '
                f' Train F1: {train_epoch_f1:.3f} | '
            )

        # Save the CSV file
        self._save_csv_file()

        # Save the model
        torch.save(
            self._model.state_dict(), 
            'models/' + self._file_name + '_model_weights.pth'
        )

        return None