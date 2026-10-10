"""Model constructors."""

from ecgcls.models.cnn import Cnn1d, parameter_count
from ecgcls.models.dilated_resnet import DilatedResNet

__all__ = ["Cnn1d", "DilatedResNet", "parameter_count"]
