from dataloader import ImageDataModule2 
#from dataloader2 import ImageDataModule2
import pytorch_lightning as pl
from trainer import CF_Explainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.profilers import PyTorchProfiler
#resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)


seed_everything(2)
data_module = ImageDataModule2(img_dir = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/training_CSV', 
                              labels_csv='/home/saptarshi/Research/encoder_decoder/physionet2017challenge/REFERENCE_.csv',  
                              batch_size=256,
                              num_workers=12,
                              merge =True,
                            )
data_module.prepare_data()

import tensorboard
model = CF_Explainer(num_classes=2,lr=3e-4,mix_up=False)
callbacks = [ModelSummary(max_depth=2),LearningRateMonitor(logging_interval="epoch"),model.checkpoint()]
logger = TensorBoardLogger(save_dir='logs')
torch.set_float32_matmul_precision('medium')

profiler = PyTorchProfiler(filename='profiler0',
                           emit_nvtx=True,
                           on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/profiler0"),
                           export_to_chrome=True,
                           trace_memoty=True,
                           schedule = torch.profiler.schedule(skip_first=3,wait=1,warmup=1,active=2)
                           )

trainer = pl.Trainer(max_epochs=1000,
                    default_root_dir='logs',
                    devices=1,
                    accelerator="gpu",
                    callbacks=callbacks,
                    precision='16-mixed',
                    logger=logger,
                    enable_model_summary=True,
                    fast_dev_run=False,
                    log_every_n_steps=20,
                    #profiler=profiler
                     )
data_module.setup()
trainer.fit(model,datamodule=data_module)
trainer.test(model,datamodule=data_module)                  