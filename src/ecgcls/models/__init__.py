"""Model constructors."""

from ecgcls.models.cnn import Cnn1d, parameter_count
from ecgcls.models.dilated_resnet import DilatedResNet
from ecgcls.models.transformer import SignalTransformer

__all__ = ["Cnn1d", "DilatedResNet", "SignalTransformer", "parameter_count"]
