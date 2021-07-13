from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.backbone.resnet import (
    BottleneckWithFixedBatchNorm,
    _make_stage,
)
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d

StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

ResNet152StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)


class StemWithFixedBatchNorm(nn.Module):
    def __init__(self, norm_func=FrozenBatchNorm2d):
        super(StemWithFixedBatchNorm, self).__init__()

        out_channels = 64

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        nn.init.kaiming_uniform_(self.conv1.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.maxpool(x)
        return x


class ResNext152(nn.Module):
    def __init__(self):
        super(ResNext152, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = StemWithFixedBatchNorm
        stage_specs = ResNet152StagesTo5
        transformation_module = BottleneckWithFixedBatchNorm

        # Construct the stem module
        self.stem = stem_module()

        # Constuct the specified ResNet stages
        num_groups = 32
        width_per_group = 8
        in_channels = 64
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = 256
        self.stages = []
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = False
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                False,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": False,
                    "deformable_groups": 1,
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(2)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def load_from_detector(self, state_dict):
        new_state_dict = dict()
        for k in state_dict:
            if k.startswith('module.backbone.body.'):
                new_state_dict[k[len('module.backbone.body.'):]] = state_dict[k]
            elif k.startswith('module.attribute.feature_extractor.head.'):
                new_state_dict[k[len('module.attribute.feature_extractor.head.'):]] = state_dict[k]

        self.load_state_dict(new_state_dict)

    def forward(self, x):
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        return x


if __name__ == "__main__":
    model = ResNext152()
    ckpt = torch.load("/home/gzx/data/scene_graph_benchmark/models/vinvl/vinvl_vg_x152c4.pth", map_location='cpu')
    model.load_from_detector(ckpt)
    model.cuda()

    from tqdm import tqdm
    import time
    total = 0
    n_test = 500
    for i in tqdm(range(n_test), total=n_test):
        image = torch.randn([1, 3, 512, 1024]).float().cuda()
        start = time.time()
        with torch.no_grad():
            _ = model(image)
            # print(_.shape)
        end = time.time() - start
        total += end

    print(total / n_test * 1000)