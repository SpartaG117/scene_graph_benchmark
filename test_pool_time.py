# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg

from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from scene_graph_benchmark.x152grid import ResNext152
from scene_graph_benchmark.coco_image_dataset import ExtractCocoDatasetTargetPath
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler

from maskrcnn_benchmark.data.transforms import build_transforms
import time
import torch.nn.functional as F

def BatchCollator(batch):
    images = batch[0].unsqueeze(0)
    hs = batch[1]
    ws = batch[2]
    return images, hs, ws


def make_data_loader(cfg, dataset, is_distributed=False):
    num_gpus = get_world_size()

    images_per_batch = cfg.TEST.IMS_PER_BATCH
    assert (
        images_per_batch % num_gpus == 0
    ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    shuffle = False if not is_distributed else True
    num_iters = None
    start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, False, images_per_gpu, num_iters, start_iter
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=BatchCollator,
    )

    return data_loader


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="sgg_configs/vgattr/vinvl_grid_x152c4.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default="models/vinvl/vinvl_vg_x152c4.pth",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # python -m torch.distributed.launch --nproc_per_node=$NGPUS  test_pool.py --config-file sgg_configs/vgattr/vinvl_grid_x152c4.yaml TEST.IMS_PER_BATCH 4
    # python -m torch.distributed.launch --nproc_per_node=3  test_pool.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 3

    # python test_pool.py --config-file sgg_configs/vgattr/vinvl_grid_x152c4.yaml TEST.IMS_PER_BATCH 1
    # python test_pool_time.py --config-file sgg_configs/vgattr/vinvl_grid_x152c4.yaml TEST.IMS_PER_BATCH 1
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=cfg.DISTRIBUTED_BACKEND, init_method="env://"
        )
        synchronize()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    model = ResNext152()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_from_detector(ckpt)
    model.to("cuda")
    model.eval()

    transforms = build_transforms(cfg, is_train=False)

    import json
    from tqdm import tqdm
    with open("/home/gzx/data/mlp_caption/test_image.json", 'r') as f:
        images_path = json.load(f)
    dataset = ExtractCocoDatasetTargetPath(images_path, transforms)
    data_loaders_val = make_data_loader(cfg, dataset, is_distributed=distributed)
    total_time = 0
    for i, data in tqdm(enumerate(data_loaders_val), total=len(data_loaders_val)):
        img = data
        # img_id = img_ids[0]
        # output_path = os.path.join(output_dir, img_id+'.npz')
        # if os.path.exists(output_path):
        #     continue
        # hw = np.array([h[0], w[0]], np.int32)
        # print(img.shape)
        img = img.cuda()
        with torch.no_grad():
            s = time.time()
            feats = model(img)
            feats = F.adaptive_avg_pool2d(feats, (6, 10)).cpu().view(2048, -1).transpose(0, 1)
            total_time += time.time() - s

    print(total_time)
    print(total_time / 5000*1000)


if __name__ == "__main__":
    main()
