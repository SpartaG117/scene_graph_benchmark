# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.engine.inference import inference
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import json
from tqdm import tqdm
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
from maskrcnn_benchmark.utils.comm import get_world_size
import logging
import time
import numpy as np


# Check if we can enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for mixed precision via apex.amp')
def BatchCollator(batch):
    images = batch[0][0].unsqueeze(0)
    hs = batch[0][1]
    ws = batch[0][2]
    img_ids = batch[0][3]
    return images, hs, ws, img_ids


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
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

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

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    # python test_box.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 1 MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2  TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True
    # python -m torch.distributed.launch --nproc_per_node=2 test_box.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2  TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


    transforms = build_transforms(cfg, is_train=False)

    from scene_graph_benchmark.coco_image_dataset import ExtractVizwizDataset
    image_dir = "/home/gzx/data/Dataset/image_caption/vizwiz/images_trainval"
    data_json = "/home/gzx/data/Dataset/image_caption/vizwiz/data_vizwiz.json"
    output_dir = "/home/gzx/data/Dataset/image_caption/vizwiz/vinvl_features"
    dataset = ExtractVizwizDataset(image_dir, data_json, transforms)

    data_loaders_val = make_data_loader(cfg, dataset, is_distributed=distributed)
    start_time = time.time()
    for i, data in enumerate(data_loaders_val):
        img = data[0]
        h = data[1]
        w = data[2]
        img_id = data[3]
        # print(img.shape, h, w, img_id)
        output_path = os.path.join(output_dir, img_id+'.npz')
        if os.path.exists(output_path):
            continue
        img = img.cuda()
        with torch.no_grad():
            prediction = model(img)
            prediction = prediction[0].to(torch.device("cpu"))
            prediction = prediction.resize((w, h))
            feats = prediction.get_field("box_features").numpy()
            # print(feats.shape)

        np.savez(output_path, feat=feats)
        mean_time = (time.time() - start_time) / (i+1)
        eta = int((len(data_loaders_val) - i - 1) * mean_time / 60)
        if i % 20 == 0:
            logger.info("save {}/{}   ETA:{}min".format(i, len(data_loaders_val), eta))
    synchronize()

if __name__ == "__main__":
    main()
