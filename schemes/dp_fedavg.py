"""Client-side Opacus DP-SGD with plaintext FedAvg aggregation.

Create the trainer once per run so client privacy accountants persist across rounds."""

from schemes.fedavg import ClientNodeClass, NetworkSimulationClass, ServerNodeClass
from training.private import make_private_trainer

__all__ = [
    "make_private_trainer",
    "ClientNodeClass",
    "ServerNodeClass",
    "NetworkSimulationClass",
]
