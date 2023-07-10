# from torchvision import models
# model = models.resnet50(pretrained=True)
#
# child_counter = 0
# for child in model.children():
#     print(" child", child_counter, "is -")
#     print(child)
#     child_counter += 1

import torch

print(torch.device('cuda'))
