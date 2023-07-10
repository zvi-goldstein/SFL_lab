import torch
import os
from Engine import Engine
from NeuralNetwork import NeuralNetwork
from utils import BScanDataset


def main():
    wd = os.getcwd()
    train_img_dir = os.path.join(wd, "IntNoInt")
    val_img_dir = os.path.join(wd, "IntNoInt_Val")

    train_data = BScanDataset(img_dir=train_img_dir)
    # MAKING TRAIN AND VALIDATION DATA THE SAME UNTIL I GET THE VALIDATION DATA ON THE COMPUTER
    val_data = BScanDataset(img_dir=train_img_dir)

    engine = Engine()

    engine.prepare_data("train", train_data, batch_size=16)
    engine.prepare_data("val", val_data, batch_size=16)

    model = NeuralNetwork(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    engine.train_model(model, criterion, optimizer, lr_scheduler, num_epochs=4)


if __name__ == "__main__":
    main()