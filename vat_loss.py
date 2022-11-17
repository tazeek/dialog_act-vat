import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=1.0, ip=1):
       
        """VAT loss
        :param xi: hyperparameter of VAT (default: 1e-6)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """

        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def _generate_vat_perturbation(self, model, pred, x, x_len):

        # Prepare random unit
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = self._l2_normalize(d)

        for _ in range(self.ip):

            # Generate the random noise
            d.requires_grad_()
            random_pert = self.xi * d

            # Perform using random perturbation
            pred_out = model(x + random_pert, x_len)
            output = F.log_softmax(pred_out, dim = 1)

            # Find adversarial distance from the KL loss
            adv_distance = F.kl_div(output, pred, reduction='batchmean')
            adv_distance.backward()

            # Update with new gradients
            d = self._l2_normalize(d.grad)
            model.zero_grad()

        return d

    def forward(self, model, x, x_len):

        model.eval()

        with torch.no_grad():
            pred = F.softmax(model(x, x_len), dim=1)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = self._l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv, x_len)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds