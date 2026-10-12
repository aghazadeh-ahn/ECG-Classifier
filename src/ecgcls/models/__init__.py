"""Model constructors."""

from ecgcls.models.atrial import AtrialCnnLstm
from ecgcls.models.beat import BeatCnn
from ecgcls.models.cnn import Cnn1d, parameter_count
from ecgcls.models.dilated_meanmax import DilatedMeanMax
from ecgcls.models.dilated_outlier import DilatedOutlier
from ecgcls.models.dilated_resnet import DilatedResNet
from ecgcls.models.dilated_rhythm import DilatedRhythmNet
from ecgcls.models.transformer import SignalTransformer

__all__ = [
    "AtrialCnnLstm",
    "BeatCnn",
    "Cnn1d",
    "DilatedMeanMax",
    "DilatedOutlier",
    "DilatedResNet",
    "DilatedRhythmNet",
    "SignalTransformer",
    "parameter_count",
]
