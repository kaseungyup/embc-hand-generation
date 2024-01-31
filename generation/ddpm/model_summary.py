import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

image_size = 64
num_channels = 128
learn_sigma = False
num_res_blocks = 2

attention_resolutions="16,8"
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))

dropout = 0.0
if image_size == 512:
    channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
elif image_size == 256:
    channel_mult = (1, 1, 2, 2, 4, 4)
elif image_size == 128:
    channel_mult = (1, 1, 2, 3, 4)
elif image_size == 64:
    channel_mult = (1, 2, 3, 4)
NUM_CLASSES = 1000
class_cond = False
use_checkpoint = False
use_fp16 = False
num_heads = 4
num_head_channels = -1
num_heads_upsample = -1
use_scale_shift_norm = True
resblock_updown = False
use_new_attention_order = False

args = create_argparser().parse_args()

dist_util.setup_dist()
logger.configure()

logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.to(dist_util.dev())
schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

logger.log("creating data loader...")
data = load_data(
    data_dir="data/nc_data_train_cmr",
    batch_size=args.batch_size,
    image_size=args.image_size,
    class_cond=args.class_cond,
)
batch, cond = next(data)


# summary(model, [(2,64,26),{}])
print(model)
