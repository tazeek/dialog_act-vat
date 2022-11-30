from models import lstm_glove, lstm_bert
from vat_loss import VATLoss
from torchmetrics import Recall
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision
from torch import nn
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix

import pickle
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
        self._base_file = params['file_name']

        # Results evaluation
        self._eval_results = self._get_results_dictionary()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model and mount to device
        self._model = self._create_model(args, params)
        self._model.to(self._device)
    
    def _metrics_evaluation(self, y_pred, y_train, device):

        # Get the predicted labels and convert to CPU
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

        # Calculate the metrics
        f1_metric = MulticlassF1Score(average = 'weighted', num_classes = 4).to(self._device)
        f1_metric = f1_metric(y_pred_tags, y_train)

        recall_metric = Recall(average = 'weighted', num_classes = 4).to(self._device)
        recall_metric = recall_metric(y_pred_tags, y_train)

        precision_metric = MulticlassPrecision(average = 'weighted', num_classes = 4).to(self._device)
        precision_metric = precision_metric(y_pred_tags, y_train)

        return precision_metric.item(), f1_metric.item(), recall_metric.item()

    def _save_csv_file(self) -> None:

        # Convert to dataframe
        df = pd.DataFrame(self._eval_results)

        # Save dataframe to CSV
        file_name = 'results/' + self._base_file + '_training_results.csv'

        df.to_csv(file_name, index=False)

        return None

    def _get_results_dictionary(self):

        return {
            'epoch': [],
            'cross_entropy_loss': [],
            'precision': [],
            'f1': [],
            'recall': []
        }

    def _create_model(self, args, params):

        model_name = args.model + '_' + args.embed

        return {
            'lstm_glove': lstm_glove.LSTM_GLove(params['embeddings']),
            'lstm_bert': lstm_bert.LSTM_Bert()
        }[model_name]

    def _reset_metrics(self):

        self._train_epoch_loss = 0
        self._train_vat_loss = 0
        self._train_epoch_f1 = 0
        self._train_epoch_prec = 0
        self._train_epoch_recall = 0

        return None

    def _print_updates(self, epoch):

        # Print every 10 epochs
        print(
            f'Epoch {epoch+0:03}: |'
            f' Train Loss: {self._train_epoch_loss:.5f} | '
            #f' VAT Loss: {(self._train_vat_loss * alpha_val)/train_set_size:.5f} | '
            f' Train Recall: {self._train_epoch_recall:.3f} | '
            f' Train F1: {self._train_epoch_f1:.3f} | '
            f' Train Precision: {self._train_epoch_prec:.3f} '
        )

        return None

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

    def start_train_glove(self):

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

            # Reset every poch
            self._reset_metrics()

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

                # Compute the loss and metrics
                train_loss = criterion(y_pred.squeeze(), y_train)
                precision, f1, recall = self._metrics_evaluation(y_pred, y_train, self._device)

                # Back propagation
                # Update for parameters and compute the updates
                train_loss.backward()
                optimizer.step()
                
                # Update for Display
                self._train_epoch_loss += train_loss.item()
                self._train_epoch_prec += precision
                self._train_epoch_f1 += f1
                self._train_epoch_recall += recall
                #self._train_vat_loss += lds.item()
            
            # Normalize results
            self._train_epoch_loss /= train_set_size
            self._train_epoch_f1 /= train_set_size
            self._train_epoch_recall /= train_set_size
            self._train_epoch_prec /= train_set_size

            # Store the results
            self._eval_results['epoch'].append(epoch)
            self._eval_results['cross_entropy_loss'].append(self._train_epoch_loss)
            self._eval_results['recall'].append(self._train_epoch_recall)
            self._eval_results['f1'].append(self._train_epoch_f1)
            self._eval_results['precision'].append(self._train_epoch_prec)

            # Print output (every 10 epochs)
            if epoch % 10 == 0:
                self._print_updates(epoch)

        # Save the CSV file
        self._save_csv_file()

        # Save the model
        torch.save(
            self._model.state_dict(), 
            'models/' + self._base_file + '_model_weights.pth'
        )

        return None

    def test_model_glove(self):
        
        # Mount model onto the GPU
        self._model.to(self._device)

        y_pred_list = []
        y_test_list = []
        
        with torch.no_grad():
            
            self._model.eval()

            for (x_original_len, x_padded, y_test) in self._test_data:

                # Convert to LongTensor
                y_test = y_test.type(torch.LongTensor)

                # Load inputs and labels onto device
                x_padded, y_test = x_padded.to(self._device), y_test.to(self._device)

                # Predict the outputs
                y_pred = self._model(x_padded, x_original_len)

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

        # Get the confusion matrix
        # Save the confusion matrix
        cm_results = confusion_matrix(y_pred_list, y_test_list)
        file_name = 'results/' + self._base_file + '_confusion_matrix.pk'
        pickle.dump(cm_results, open(file_name, "wb"))
        print(cm_results)

        print('\n')
        print(classification_report(y_test_list, y_pred_list))

        return None