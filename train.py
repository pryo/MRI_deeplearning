import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import json
from datetime import datetime
from utility import getTestAcc
import time
import os
import copy
import torch.utils as utils
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataset import *

import densenet as MD

import pickle
import torchvision

data_dir = r"C:\Users\wzuo\Developer\ML for APT\data"
gz_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\gz_list.p"
kki_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\kki_list.p"
ROI_log_path = r"C:\Users\wzuo\Developer\ML for APT\APT\1D_APT_ROI_Log.csv"
gz_list=pickle.load( open( gz_list_path, "rb" ) )
kki_list=pickle.load( open( kki_list_path, "rb" ) )
param_dict = {}
param_dict['training_file'] = os.path.basename(__file__)
param_dict['memo'] = 'fixed the image feeding attempt 2'
img_transform = transforms.Compose(
    [transforms.Resize([224,224]),
        transforms.ToTensor()
        ])
#param_dict['image_transform'] = img_transform.__str__()
#todo need to change normalization coefficient

dataset_switch = 1

apt_dataset = DualChannelAPTDataset(data_dir,gz_list,kki_list,ROI_log_path,
                         truth_path=r"C:\Users\wzuo\Developer\ML for APT\idh.xlsx",
                         transform=img_transform,switch=dataset_switch)
param_dict['dataset_switch'] = dataset_switch
param_dict['ppms'] = apt_dataset.getPPMs()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_train = len(apt_dataset)
indices = list(range(num_train))
val_split = 10
test_split = 10
param_dict['val_split'] = val_split
param_dict['test_split'] = test_split
# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=val_split, replace=False)
train_idx = list(set(indices) - set(validation_idx))
test_idx = np.random.choice(train_idx,size=test_split,replace=False)
train_idx = list(set(train_idx)-set(test_idx))


#define our samplers -- we use a SubsetRandomSampler because it will return
# a random subset of the split defined by the given indices without replaf
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)
test_sampler = SubsetRandomSampler(test_idx)

param_dict['train_batch'] =10
param_dict['val_batch'] =10
param_dict['test_batch'] = 1
train_loader = torch.utils.data.DataLoader(apt_dataset,
                batch_size=param_dict['train_batch'], sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(apt_dataset,
                batch_size=param_dict['val_batch'], sampler=validation_sampler)
test_loader = torch.utils.data.DataLoader(apt_dataset,
                batch_size=param_dict['test_batch'], sampler=test_sampler)



def train_model(modelLocal,modelGlobal,extClassifier,train_loader,validation_loader,device,criterion, optimizer, scheduler,param_dict, num_epochs=25):

    since = time.time()

    #best_model_wts = copy.deepcopy(modelGlobal.state_dict())
    #best_patchModel_wts = copy.deepcopy(modelLocal.state_dict())
    best_classifier_wts = copy.deepcopy(externalClassifier.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                modelLocal.train()
                modelGlobal.train()
                extClassifier.train()
                # Set model to training mode
            else:
                modelLocal.eval()
                modelGlobal.eval()
                extClassifier.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if phase =='train':
                for batch_idx, sample in enumerate(train_loader):
                    images = sample['image'].to(device)
                    patchs = sample['patch'].to(device)
                    ages = sample['age'].to(device)
                    labels = sample['label'].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputLocal = modelLocal(patchs)
                        outputGlobal = modelGlobal(images)
                        interim = torch.cat((outputGlobal, outputLocal), 1)
                        interim = torch.cat((ages.unsqueeze(1), interim), 1)
                        outputs = externalClassifier(interim)
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
                    patchs = sample['patch'].to(device)
                    ages = sample['age'].to(device)
                    labels = sample['label'].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputLocal = modelLocal(patchs)
                        outputGlobal = modelGlobal(images)
                        interim = torch.cat((outputGlobal, outputLocal), 1)
                        interim = torch.cat((ages.unsqueeze(1), interim), 1)
                        outputs = externalClassifier(interim)
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
                epoch_loss = running_loss / (len(apt_dataset)-val_split)
                epoch_acc = running_corrects.double() / (len(apt_dataset)-val_split)
            else:
                epoch_loss = running_loss / val_split
                epoch_acc = running_corrects.double() / val_split

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                # best_patchModel_wts = copy.deepcopy(patchModel.state_dict())
                best_classifier_wts = copy.deepcopy(extClassifier.state_dict())
                param_dict['val_acc'] = 'val_acc {:4f}'.format(best_acc)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #patchModel.load_state_dict(best_patchModel_wts)
    extClassifier.load_state_dict(best_classifier_wts)
    model_serial = str(datetime.now().timestamp())
    # torch.save(model.state_dict(),
    #            os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.model'))
    # torch.save(patchModel.state_dict(),
    #            os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.patchModel'))
    torch.save(extClassifier.state_dict(),
               os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.clsmodel'))
    with open(os.path.join(r'C:\Users\wzuo\Developer\ML for APT\models', model_serial + '.json'), 'w') as fp:
        json.dump(param_dict, fp)

    return extClassifier


model_global = MD.densenet201(pretrained=True)
model_local = MD.densenet169(pretrained=True)

param_dict['model_global'] = 'densenet201'
param_dict['model_local'] = 'densenet169'

# todo change architecture from here
#patch_model = MD.densenet121(pretrained=True)
#patch_model = MD.SimpleNet()
#param_dict['patch_base'] = 'densenet121'

for param in model_global.parameters():
    param.requires_grad = False

for param in model_local.parameters():
    param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs_global = model_global.classifier.in_features

num_ftrs_local = model_local.classifier.in_features

externalClassifier =nn.Sequential(
    nn.Linear(num_ftrs_global+num_ftrs_local+1, 256),
    nn.ReLU(),
    nn.Linear(256,2)
)
# externalClassifier =nn.Sequential(
#     nn.Linear(num_ftrs_global+num_ftrs_local+1, 2)
#
# )
param_dict['classifire_shape'] = 'global+local+1,256,2'
#model_conv.ageFC = nn.Linear(1,32)
model_global = model_global.to(device)
model_local = model_local.to(device)
externalClassifier = externalClassifier.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
param_dict['optimizer_lr'] = 0.001
param_dict['optimizer_momentum'] = 0.9
optimizer_external_classifier = optim.SGD(externalClassifier.parameters(), lr=param_dict['optimizer_lr'],
                           momentum=param_dict['optimizer_momentum'])


param_dict['lr_scheduler_step_size'] = 7
param_dict['lr_scheduler_gamma'] = 0.1
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_external_classifier, step_size=param_dict['lr_scheduler_step_size'],
                                       gamma=param_dict['lr_scheduler_gamma'])

param_dict['epochs'] = 30
model_conv = train_model(model_local,model_global,externalClassifier,
                         train_loader,
                         validation_loader,
                         device,
                         criterion,
                         optimizer_external_classifier,
                         exp_lr_scheduler,
                         param_dict,
                         num_epochs=param_dict['epochs'])

test_acc = getTestAcc(externalClassifier,test_loader,model_local,model_global,device,test_split)
print('test_acc:{:4f}'.format(test_acc))