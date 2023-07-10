import time
import copy
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


class Engine():

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset_sizes = {x: None for x in ['train', 'val']}
        self.datasets = {x: None for x in ['train', 'val']}
        self.dataLoaders = {x: None for x in ['train', 'val']}

    def prepare_data(self, phase: str, dataset, batch_size=64, shuffle=True):
        self.dataset_sizes[phase] = len(dataset)
        self.dataLoaders[phase] = DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):

        model.to(self.device)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                with tqdm(self.dataLoaders[phase], unit="batch") as dataLoader:

                    for inputs, labels in dataLoader:
                        dataLoader.set_description(f"Epoch {epoch}")

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            # This if statement is a an easy but not very good fix for case where final batch is size 1.
                            if len(outputs.size()) == 2:
                                _, preds = torch.max(outputs, 1)
                            else:
                                _, preds = torch.max(outputs, 0)
                                print(outputs)
                            # ADDED .float()
                            loss = criterion(outputs, labels.float())

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        if len(labels.size()) == 2:
                            _, ground_truth = torch.max(labels, 1)
                        else:
                            _, ground_truth = torch.max(labels, 0)
                            print(labels)
                        running_corrects += torch.sum(preds == ground_truth)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
