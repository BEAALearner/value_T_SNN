import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import neuron, layer, surrogate
from spikingjelly.activation_based.neuron import ParametricLIFNode

input_dim = 2 * 34 * 34

class ParametricLIFNodeImprovedTau(neuron.BaseNode):
    def __init__(self, init_tau=1.1, tau_min=1e-2, tau_max=1e2):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.log_tau_raw = nn.Parameter(torch.tensor(math.log(init_tau)))

    @property
    def tau(self):
        return torch.exp(self.log_tau_raw).clamp(self.tau_min, self.tau_max)

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class LeakGatedLIFNode(neuron.BaseNode):
    def __init__(self, num_neurons, tau=2.0):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau = tau
        self.leak_gate = nn.Sequential(            nn.Linear(1, 1),            nn.Tanh()        )

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        v_reshaped = self.v.view(-1, 1)
        gated_leak = self.leak_gate(v_reshaped).view_as(self.v)  # Reshape back
        self.v = self.v + (x - gated_leak) / self.tau


class AttentionLIFNode(neuron.BaseNode):
    def __init__(self, num_neurons):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.w_v = nn.Parameter(torch.randn(num_neurons) * 0.1)
        self.w_x = nn.Parameter(torch.randn(num_neurons) * 0.1)
        self.b = nn.Parameter(torch.zeros(num_neurons))

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        alpha = torch.sigmoid(self.w_v * self.v + self.w_x * x + self.b)
        self.v = alpha * self.v + (1.0 - alpha) * x

# 2. Adaptive LIF (AdLIF) Neuron
class AdLIFNode(neuron.BaseNode):
    def __init__(self, num_neurons, tau_init=2.0, tau_min=0.1, tau_max=10.0):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau = nn.Parameter(torch.full((num_neurons,), tau_init))
        self.tau_min = tau_min
        self.tau_max = tau_max

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        tau_clamped = self.tau.clamp(self.tau_min, self.tau_max)
        self.v = self.v + (x - (self.v - self.v_reset)) / tau_clamped


# 3. GLIF (Gated LIF) Neuron - More complex version
class GLIFNode(neuron.BaseNode):
    def __init__(self, num_neurons, tau=2.0):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau = tau
        self.gate = nn.Sequential(            nn.Linear(2, 4),            nn.Tanh(),            nn.Linear(4, 1),            nn.Sigmoid()        )
    
    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        inputs = torch.stack([self.v, x], dim=-1)
        gate = self.gate(inputs).squeeze(-1)
        self.v = self.v + (gate * x - (self.v - self.v_reset)) / self.tau


class FixedAlphaLIFNode(neuron.BaseNode):
    def __init__(self, num_neurons, fixed_alpha=0.5): # You can make fixed_alpha an argument
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.fixed_alpha = fixed_alpha # Store the fixed alpha value

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        # Instead of learning alpha, use the fixed_alpha
        alpha = self.fixed_alpha
        self.v = alpha * self.v + (1.0 - alpha) * x



class ExpIFNode(neuron.BaseNode):
    def __init__(self, tau=2.0, delta_T=1.0, v_thresh=1.0, v_reset=0.0):
        super().__init__(v_threshold=v_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.tau = tau
        self.delta_T = delta_T

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.v += (x + self.delta_T * torch.exp((self.v - self.v_threshold) / self.delta_T)) / self.tau



class ResLIFNode(neuron.BaseNode):
    def __init__(self, tau=2.0, alpha=0.3):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau = tau
        self.alpha = alpha

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        residual = self.alpha * x
        self.v += (x - (self.v - self.v_reset)) / self.tau + residual


class MTC_LIFNode(neuron.BaseNode):
    def __init__(self, num_neurons, tau_base=2.0):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau_base = tau_base
        self.tau_modulator = nn.Sequential(
            nn.Linear(1, 1),  # simple modulation based on v
            nn.Tanh()
        )

    def neuronal_charge(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        # Modulate tau based on voltage
        tau_mod = self.tau_modulator(self.v.view(-1, 1)).view_as(self.v)
        tau_eff = self.tau_base * (1.0 + tau_mod)
        self.v = self.v + (x - (self.v - self.v_reset)) / tau_eff.clamp(min=0.1)



class ModularSNN(nn.Module):
    def __init__(self, neuron_type="plif"):
        super().__init__()
        self.neuron_type = neuron_type
        self.fc1 = layer.Linear( input_dim, 128, bias=False)
        self.lif1 = self.get_neuron(128)
        self.fc2 = layer.Linear(128, 10, bias=False)
        self.lif2 = self.get_neuron(10)

    def get_neuron(self, num_neurons):
        if self.neuron_type == "lif":
            return neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "plif":
            return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == "attention":
            return AttentionLIFNode(num_neurons=num_neurons)
        elif self.neuron_type == "fixed":
            return FixedAlphaLIFNode(num_neurons=num_neurons)
        elif self.neuron_type == "adlif":
            return AdLIFNode(num_neurons=num_neurons)
        elif self.neuron_type == "glif":
            return GLIFNode(num_neurons=num_neurons)
        elif self.neuron_type == "mtc_lif":
            return MTC_LIFNode(num_neurons=num_neurons)
        elif self.neuron_type == "improved_tau":
            return ParametricLIFNodeImprovedTau()
        elif self.neuron_type == "expif":
            return ExpIFNode()
        elif self.neuron_type == "reslif":
            return ResLIFNode()
        elif self.neuron_type == "qif":
            return neuron.QIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "eif":
            return neuron.EIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "clif":
            return CLIFNode(c_decay=0.5, v_decay=0.75, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "adaptive":
            return neuron.AdaptBaseNode(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan(), out_features=num_neurons)
        elif self.neuron_type == "gated":
            return neuron.GatedLIFNode(T=5, inplane=num_neurons, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "mpbn":
            return neuron.MPBNLIFNode(out_features=num_neurons, surrogate_function=surrogate.ATan())
        elif self.neuron_type == "izi":
            return neuron.IzhikevichNode(tau=2.0, surrogate_function=surrogate.ATan())

        else:
            raise ValueError(f"Unsupported neuron type: {self.neuron_type}")


    def forward(self, x):
        x_flattened = x.flatten(1)
        h1 = self.fc1(x_flattened)
        s1 = self.lif1(h1)
        h2 = self.fc2(s1)
        s2 = self.lif2(h2)
        return s2


def get_model(neuron_type):
    return ModularSNN(neuron_type=neuron_type)
