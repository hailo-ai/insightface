import argparse
import logging
import sys

import torch
from torch import distributed

from backbones import get_model
from utils.utils_callbacks import CallBackVerification
from utils.utils_config import get_config


class DummySummaryWriter:
    def add_scalar(*args, **kwargs):
        pass


def init_logging(rank):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_stream.setFormatter(formatter)


def main(args):
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,

    )
    init_logging(rank=rank)
    cfg = get_config(args.config)
    model = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    model.load_state_dict(torch.load(args.weight))
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=DummySummaryWriter
    )

    with torch.no_grad():
        callback_verification(1, model)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument('--weight', type=str)
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
