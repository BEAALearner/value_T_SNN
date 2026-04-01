import os
import torch
import pandas as pd


class SNNLogger:
    def __init__(self, model_name, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.logs = { 'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': [],
            'spike_rate': [], 'tau_lif1': [], 'tau_lif2': [], 'epoch_time': [], }

        self.fn = f'{model_name}.csv'

    def update_train(self, epoch, loss, acc, spike_tensor, model):
        self.logs['epoch'].append(epoch)
        self.logs['train_loss'].append(loss)
        self.logs['train_acc'].append(acc)
        self.logs['spike_rate'].append(spike_tensor.mean().item())

        tau1 = self._get_tau(model.lif1)
        tau2 = self._get_tau(model.lif2)
        self.logs['tau_lif1'].append(tau1)
        self.logs['tau_lif2'].append(tau2)


    def update_test(self, acc):
        self.logs['test_acc'].append(acc)
        
    def update_epoch_time(self, seconds):
        self.logs['epoch_time'].append(seconds)

    def export_csv(self, filename= None):
        max_len = max(len(v) for v in self.logs.values())
        filename = self.fn
        for k in self.logs:
            while len(self.logs[k]) < max_len:
                self.logs[k].append(None)
        df = pd.DataFrame(self.logs)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        print(f"📄 Metrics exported to {path}")

    def save_model(self, model, filename= None):
        print("")

    def _get_tau(self, neuron):
        """Handle LogParametricLIFNode's special case"""
        # First check for standard tau attributes
        if hasattr(neuron, 'tau'):
            tau = neuron.tau
            return tau.mean().item() if isinstance(tau, torch.Tensor) else float(tau)
        # Special case for LogParametricLIFNode
        if hasattr(neuron, 'log_tau_raw'):
            return torch.exp(neuron.log_tau_raw).mean().item()
        # Fallback checks
        for attr in ['tau_m', 'tau_membrane', 'membrane_time_constant']:
            if hasattr(neuron, attr):
                val = getattr(neuron, attr)
                return val.item() if isinstance(val, torch.Tensor) else float(val)

        return None
