from models import lstm_glove
from vat_loss import VATLoss

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