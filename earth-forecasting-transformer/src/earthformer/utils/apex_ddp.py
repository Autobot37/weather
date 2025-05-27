from torch.nn.parallel import DistributedDataParallel
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
)

def unwrap_lightning_module(wrapped_model):
    model = wrapped_model
    if isinstance(model, DistributedDataParallel):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPStrategy(DDPStrategy):
    def _setup_model(self, model):
        return DistributedDataParallel(
            model, 
            find_unused_parameters=True,
            gradient_as_bucket_view=True  # This may help with the stride warning
        )

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)