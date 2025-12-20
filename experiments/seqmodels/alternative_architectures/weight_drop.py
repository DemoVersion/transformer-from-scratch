import torch.nn as nn
import torch.nn.functional as F


class WeightDrop(nn.Module):
    """
    Applies DropConnect (Weight Dropout) to the weights of a specified module.
    Crucial for AWD-LSTM.
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout

        # Register a hook for each weight we want to drop
        for name_w in self.weights:
            # 1. Store the original weight
            w = getattr(module, name_w)
            del module._parameters[name_w]
            module.register_parameter(name_w + "_raw", nn.Parameter(w.data))

            # 2. Register a forward pre-hook to apply dropout
            self.module.register_forward_pre_hook(self.forward_pre_hook)

    def _setweights(self, name_w):
        """Applies dropout mask to the raw weight."""
        raw_w = getattr(self.module, name_w + "_raw")

        # Apply dropout (this is DropConnect)
        w = F.dropout(raw_w, p=self.dropout, training=self.training)

        # Set the module's actual weight parameter to the dropped version
        setattr(self.module, name_w, w)

    def forward_pre_hook(self, module, inputs):
        """Called just before the LSTM forward pass runs."""
        for name_w in self.weights:
            self._setweights(name_w)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
