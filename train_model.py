from vat_loss import VATLoss
from torchmetrics import Recall, ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision
from torch import nn
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix

import pickle
import pandas as pd
import torch

class Model():

    def __init__(self, args, logger):

        # Hyperparams
        self._lr = args['learning_rate']
        self._epochs = args['epochs']
        self._alpha_val = args['alpha_val']

        # Data Generators
        self._train_data = args['training']
        self._test_data = args['test']
        self._unlabeled_data = args['unlabeled']

        # Others
        self._base_file = args['file_name']
        self._logger = logger
        self._embed = args['embed']
        self._train_size = len(self._train_data)

        # Results evaluation
        self._eval_results = self._get_results_dictionary()
        self._metric_evaluator = self._initalize_metrics()

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create model and mount to device
        self._model = args['model']
        self._model.to(self._device)
    
    def _initalize_metrics(self):

        return {
            'f1': MulticlassF1Score(average = 'weighted', num_classes = 4).to(self._device),
            'precision': MulticlassPrecision(average = 'weighted', num_classes = 4).to(self._device),
            'recall': Recall(average = 'weighted', num_classes = 4).to(self._device),
        }

    def _intialize_hyperparam_loss(self):

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr = self._lr
        )

        self._loss_func = nn.CrossEntropyLoss()

        return None

    def _batch_type(self):

        return {
            'bert': self._bert_batch(),
            'glove': self._glove_batch()
        }[self._embed]
    
    def _save_csv_file(self, file_name) -> None:

        # Convert to dataframe
        df = pd.DataFrame(self._eval_results)

        # Save dataframe to CSV
        file_name = 'results/' + self._base_file + f'_{file_name}.csv'
        df.to_csv(file_name, index=False)

        return None

    def _get_results_dictionary(self) -> dict:

        return {
            'epoch': [],
            'cross_entropy_loss': [],
            'precision': [],
            'f1': [],
            'recall': []
        }

    def _reset_metrics(self) -> None:

        self._train_epoch_loss = 0
        self._train_vat_loss = 0
        self._train_epoch_f1 = 0
        self._train_epoch_prec = 0
        self._train_epoch_recall = 0

        return None

    def _print_updates(self, epoch) -> None:

        print(
            f'Epoch {epoch+0:03}: |'
            f' Train Loss: {self._train_epoch_loss:.5f} | '
            f' Train Recall: {self._train_epoch_recall:.5f} | '
            f' Train F1: {self._train_epoch_f1:.5f} | '
            f' Train Precision: {self._train_epoch_prec:.5f} '
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

    def _metrics_evaluation(self, y_pred_logits, y_train):

        # Logits -> Probability distribution
        y_pred_distribution = torch.log_softmax(y_pred_logits, dim = 1)

        # Probability distribution -> Most likely tag (and the indices)
        y_pred_tags = torch.argmax(y_pred_distribution, dim = 1)

        # Calculate the metrics
        # TODO: Initialize the metrics once, instead of so many calls (speed up)
        f1_metric = MulticlassF1Score(average = 'weighted', num_classes = 4).to(self._device)
        f1_metric = f1_metric(y_pred_tags, y_train)

        recall_metric = Recall(average = 'weighted', num_classes = 4).to(self._device)
        recall_metric = recall_metric(y_pred_tags, y_train)

        precision_metric = MulticlassPrecision(average = 'weighted', num_classes = 4).to(self._device)
        precision_metric = precision_metric(y_pred_tags, y_train)

        return precision_metric.item(), f1_metric.item(), recall_metric.item()


    def _compute_loss_results(self, y_pred_logits, y_train) -> None:

        train_loss = self._loss_func(y_pred_logits, y_train)
        precision, f1, recall = self._metrics_evaluation(y_pred_logits, y_train)

        # Back propagration and update for parameters
        train_loss.backward()
        self._optimizer.step()

        # Update metrics
        self._train_epoch_loss += train_loss.item()
        self._train_epoch_prec += precision
        self._train_epoch_f1 += f1
        self._train_epoch_recall += recall
        #self._train_vat_loss += lds.item()

        return None

    def _normalize_and_store(self, epoch) -> None:

        # Normalize results
        self._train_epoch_loss /= self._train_set_size
        self._train_epoch_f1 /= self._train_set_size
        self._train_epoch_recall /= self._train_set_size
        self._train_epoch_prec /= self._train_set_size

        # Store the results
        self._eval_results['epoch'].append(epoch)
        self._eval_results['cross_entropy_loss'].append(self._train_epoch_loss)
        self._eval_results['recall'].append(self._train_epoch_recall)
        self._eval_results['f1'].append(self._train_epoch_f1)
        self._eval_results['precision'].append(self._train_epoch_prec)

        return None

    def start_training(self):

        self._logger.info('Initialize hyperparameters and loss functions')
        
        train_batch = self._batch_type()
        self._intialize_hyperparam_loss()

        self._model.train()

        for epoch in range(1, self._epochs + 1):

            # Reset every epoch
            self._reset_metrics()

            # Start Training
            train_batch()

            # Normalize and record the results every epoch
            self._normalize_and_store(epoch)

            # Print every epoch or every X epoch
            self._print_updates(epoch)

        # Save the CSV file
        self._save_csv_file('training_results')

        # Save the model
        torch.save(
            self._model.state_dict(), 
            'models/' + self._base_file + '_model_weights.pth'
        )
        
        return None

    def _glove_batch(self):

        for batch_data in self._train_data:

            # Unpack the dictionary and mount to device
            x_train = batch_data['features'].to(self._device)
            x_original_length = batch_data['original_length'].to(self._device)

            y_train = batch_data['labels']
            y_train = y_train.type(torch.LongTensor)
            y_train = y_train.to(self._device)

            # Get the model outputs
            self._optimizer.zero_grad()
            y_pred_logits = self._model(x_train, x_original_length)

            # Compute the loss and metrics
            self._compute_loss_results(y_pred_logits, y_train)

        return None

    def _bert_batch(self):

        for batch_data in self._train_data:

            # Unpack the data from the batch
            x_train = batch_data['features'].to(self._device)

            y_train = batch_data['labels']
            y_train = y_train.type(torch.LongTensor)
            y_train = y_train.to(self._device)

            # Get the model outputs
            self._optimizer.zero_grad()
            y_pred_logits = self._model(x_train)

            # Compute the loss and metrics
            self._compute_loss_results(y_pred_logits, y_train)

        return None

    def test_model_glove(self):

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
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=4)
        confusion_matrix_results = confusion_matrix(y_pred_list, y_test_list)
        print(confusion_matrix_results)

        # Save the confusion matrix
        file_name = 'results/' + self._base_file + '_confusion_matrix.pth'
        torch.save(confusion_matrix_results, file_name)

        print('\n')
        print(classification_report(y_test_list, y_pred_list))

        return None