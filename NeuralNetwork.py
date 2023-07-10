import torch
import torch.nn as nn
from torchvision import models


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(NeuralNetwork, self).__init__()

        self.model = models.resnet50(pretrained=True)


    # For freezing layers:

        # child_counter = 0
        # for child in self.model.children():
        #     if child_counter < 7:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     elif child_counter == 6:
        #         children_of_child_counter = 0
        #         for children_of_child in child.children():
        #             if children_of_child_counter < 1:
        #                 for param in children_of_child.parameters():
        #                     param.requires_grad = False
        #             else:
        #             children_of_child_counter += 1

        #     else:
        #         print("child ",child_counter," was not frozen")
        #     child_counter += 1

        num_fc_in = self.model.fc.in_features
        self.input_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_layers = nn.Sequential(*list(self.model.children())[1:-1])
        self.intermediate_layer = nn.Linear(num_fc_in, 512)
        self.output_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.resnet_layers(x)
        x = torch.squeeze(x)
        x = self.intermediate_layer(x)
        x = self.output_layer(x)
        return x