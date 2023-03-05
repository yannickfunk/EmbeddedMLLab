from nni.compression.pytorch.pruning import 

import nni
from nni.compression.pytorch import LightningEvaluator
from nni.compression.pytorch.speedup import ModelSpeedup

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from nni.compression.pytorch.pruning import LotteryTicketPruner, AutoCompressPruner
from nni.compression.pytorch.utils import count_flops_params
from nni.algorithms.compression.v2.pytorch.utils.pruning import compute_sparsity



config_list = [{
    'op_types': ['Conv2d'],
    'total_sparsity': 0.5
},{
    'exclude': True,
    'op_names': [],
    'op_types': []
}]

def build_evaluator(trainer: pl.Trainer, datamodule: pl.LightningDataModule):

    trainer = nni.trace(trainer)

    evaluator = LightningEvaluator(trainer=trainer, data_module=datamodule)

    return evaluator, trainer, datamodule


## Post training pruning

def auto_prune(model: pl.LightningModule, trainer: pl.Trainer, datamodule: pl.LightningDataModule, config_list: list) -> pl.LightningModule:

    evaluator, trainer, _ = build_evaluator(trainer=trainer, datamodule=datamodule)

    pruner = AutoCompressPruner(model=model, config_list=config_list, total_iteration=25, admm_params={'evaluator': evaluator, 'iterations': 7, 'epochs': 3}, sa_params={'evaluator': evaluator}, log_dir='./checkpoints', keep_intermediate_result=True, evaluator=evaluator)

    pruner.compress()

    _, model, masks, _, _ = pruner.get_best_result()

    ModelSpeedup(model, evaluator.get_dummy_input(), masks).speedup_model()

    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    return model


def lottery_pruning(model: pl.LightningModule, trainer: pl.Trainer, datamodule: pl.LightningDataModule, config_list: list) -> pl.LightningModule:
    
    evaluator, trainer, _ = build_evaluator(trainer=trainer, datamodule=datamodule)


    pruner = LotteryTicketPruner(model=model, config_list=config_list, pruning_algorithm=)  

    pruner.compress()

    _, model, masks, _, _ = pruner.get_best_result()

    ModelSpeedup(model, evaluator.get_dummy_input(), masks).speedup_model()

    return model
