from models import lstm_glove
from vat_loss import VATLoss
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

import pandas as pd
import torch

class Model():

    def __init__(self, args):

        # Hyperparams
        self._lr = 0.001
        self._epochs = 10
        self._alpha_val = 0.01

        # Create model
        self._model = self._create_model(args)

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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

    def _save_csv_file(self, data, base_file) -> None:

        # Convert to dataframe
        df = pd.DataFrame(data)

        # Save dataframe to CSV
        file_name = 'results/' + base_file + '_training_results.csv'
        df.to_csv(file_name, index=False)

        return None

    def _get_results_dictionary(self):

        return {
            'epoch': [],
            'cross_entropy_loss': [],
            'accuracy': [],
            'f1': []
        }

    def _create_model(self, args):

        model = lstm_glove.LSTM_GLove()

        return model

    def _vat_loss_calculation(self, model, device, validation_data):

        # For VAT Loss
        x_original_len_val, x_padded_val, _ = next(iter(validation_data))
        x_padded_val = x_padded_val.to(device)

        vat_loss = VATLoss()
        lds = vat_loss(model, x_padded_val, x_original_len_val)

        return lds