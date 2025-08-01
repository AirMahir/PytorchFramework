import torch
import timm
import segmentation_models_pytorch as smp
from torch.amp import autocast

# model = timm.create_model('convnextv2_nano', num_classes=2).cuda()
model = smp.Unet(
        encoder_name="timm-efficientnet-b0",
        encoder_weights="noisy-student",
        in_channels=3,
        classes=2
    ).cuda()
image = torch.randint(10, (16, 3, 256, 256)).to(torch.float32).cuda()

model.eval()
with autocast('cuda'):
    with torch.set_grad_enabled(False):
        out = model(image)
print(out.mean())

model.eval()
with torch.set_grad_enabled(False):
    out = model(image)
print(out.mean())

model.eval()
with torch.set_grad_enabled(False):
    out = model(image)
    #out = torch.argmax(out, dim=1)
print(out.mean())