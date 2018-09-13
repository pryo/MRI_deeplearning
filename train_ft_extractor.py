import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import json
from datetime import datetime
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.utils as utils
from torch.utils.data.sampler import SubsetRandomSampler
from loader import *
from densenet import *
import pickle
data_dir = r"C:\Users\wzuo\Developer\ML for APT\data"
gz_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\gz_list.p"
kki_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\kki_list.p"
gz_list=pickle.load( open( gz_list_path, "rb" ) )
kki_list=pickle.load( open( kki_list_path, "rb" ) )
param_dict = {}
param_dict['training_file'] = os.path.basename(__file__)


img_transform = transforms.Compose(
    [transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

dataset_switch = 1
apt_dataset = APTDataset(data_dir,gz_list,kki_list,
                         truth_path=r"C:\Users\wzuo\Developer\ML for APT\idh.xlsx",
                         transform=img_transform,switch=dataset_switch)
param_dict['dataset_switch'] = dataset_switch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_train = len(apt_dataset)
indices = list(range(num_train))
split = 20
param_dict['split'] = split
# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))



## define our samplers -- we use a SubsetRandomSampler because it will return
## a random subset of the split defined by the given indices without replaf
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)
param_dict['train_batch'] =25
param_dict['val_batch'] =20
train_loader = torch.utils.data.DataLoader(apt_dataset,
                batch_size=param_dict['train_batch'], sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(apt_dataset,
                batch_size=param_dict['val_batch'], sampler=validation_sampler)



def train_model(model,train_loader,validation_loader,device,criterion, optimizer, scheduler,param_dict, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if phase =='train':
                for batch_idx, sample in enumerate(train_loader):
                    images = sample['image'].to(device)
                    ages = sample['age'].to(device)
                    labels = sample['label'].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images,ages)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if phase =='val':
                for batch_idx, sample in enumerate(validation_loader):
                    images = sample['image'].to(device)
                    ages = sample['age'].to(device)
                    labels = sample['label'].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images,ages)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / (len(apt_dataset)-split)
                epoch_acc = running_corrects.double() / (len(apt_dataset)-split)
            else:
                epoch_loss = running_loss / split
                epoch_acc = running_corrects.double() / split

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                param_dict['val_acc'] = 'val_acc {:4f}'.format(best_acc)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_serial = str(datetime.now().timestamp())
    torch.save(model.state_dict(),
               os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.model'))

    with open(os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.json'), 'w') as fp:
        json.dump(param_dict, fp)

    return model


model_conv = densenet169(pretrained=True)
param_dict['base'] = 'densenet169'
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classifier.in_features
model_conv.classifier = nn.Linear(num_ftrs+32, 2) # and age vector
#model_conv.ageFC = nn.Linear(1,32)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
param_dict['optimizer_lr'] = 0.001
param_dict['optimizer_momentum'] = 0.9
optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=param_dict['optimizer_lr'],
                           momentum=param_dict['optimizer_momentum'])


param_dict['lr_scheduler_step_size'] = 7
param_dict['lr_scheduler_gamma'] = 0.1
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=param_dict['lr_scheduler_step_size'],
                                       gamma=param_dict['lr_scheduler_gamma'])

param_dict['epochs'] = 25
model_conv = train_model(model_conv,
                         train_loader,
                         validation_loader,
                         device,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         param_dict,

                         num_epochs=param_dict['epochs'])