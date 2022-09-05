import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.models.quantization.mobilenetv2 import mobilenet_v2 as mobilenetv2_q


class MobileNetV2Backbone(torch.nn.Module):
    def __init__(self, pretrained: bool = False, device: str = "cuda"):
        super().__init__()
        self.device = device
        # self.model = mobilenet_v2(pretrained=pretrained).eval().to(device)
        self.model = mobilenetv2_q(pretrained=pretrained).eval().to(device)

    def exec(self, x):
        start_time = time.time()
        if isinstance(x, str):
            x = Image.open(x)

        if isinstance(x, torch.Tensor):
            preprocess = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(224),
            ])
            x = preprocess(x).to(self.device)
            with torch.no_grad():
                y = self.model(x)
            inference_time = time.time() - start_time
            return y, inference_time






