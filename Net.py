import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.base import BaseModel, load_dualpath_model
from semseg.unet_decoder import DTrCHead


class Net(BaseModel):
    def __init__(self, backbone: str = 'CMNeXt-B0', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar']) -> None:
        super().__init__(backbone, num_classes, modals)
        self.decode_head = DTrCHead([64, 128, 320, 512],num_classes) #DTrC
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x) # 编码器阶段
        y = self.decode_head(y) # 解码器阶段
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)

