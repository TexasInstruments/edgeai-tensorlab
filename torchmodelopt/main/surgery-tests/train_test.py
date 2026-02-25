from functools import partial
from utils import StochasticBottleneck, Bottleneck # type: ignore

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, StepLR, ChainedScheduler, CosineAnnealingLR, OneCycleLR
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.v2 as transforms
from torchvision.transforms._presets import ImageClassification
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, default_collate

import mlflow
import optuna


class SimpleModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=(7,7), stride=2)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d((5,5))

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = torch.flatten(self.pool(x), 1)
        return x

def count_params(model):
    model.train()
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total 

def construct_resnet_model(params=None):
    """
        params has model hyperparameters
    """
    if params is None:
        params = {
            'weights': 'none',
            'strides': [2,2,2],
            'half-head': False,
            'first-kernel': (9,9)
            }
    weights = params.get("weights", "v2")
    if weights == 'simple':
        return SimpleModel(out_channels=64)
    if weights == "v1":
        model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
            # Replace final layer 'fc' of model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 100)
    elif weights == "v2":
        model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
                # Replace final layer 'fc' of model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 100)
    else:
        model = resnet50(num_classes=100)
        strides = params.get('strides', [2,2,2])
        model.conv1 = nn.Conv2d(3, 64, kernel_size=params.get('first-kernel',(3,3)), stride=(1,1), padding=(1,1), bias=False)
        if params.get('first-avg-pool',False):
            model.maxpool = nn.AvgPool2d(kernel_size=params.get('first-max-pool',3), stride=2, padding=1)
        else:
            model.maxpool = nn.MaxPool2d(kernel_size=params.get('first-max-pool',3), stride=2, padding=1)
        model.inplanes = 256
        model.layer2 = model._make_layer(Bottleneck, 128, 4, stride=strides[0], dilate=False)
        model.layer3 = model._make_layer(Bottleneck, 256, 6, stride=strides[1], dilate=False)
        if not params.get('half-head',False):
            # torch's normal size
            model.layer4 = model._make_layer(Bottleneck, 512, 3, stride=strides[2], dilate=False)
        else:
            # half the channels, so that FC layer has 1024 in channels and not 2048
            model.layer4 = model._make_layer(Bottleneck, 256, 3, stride=strides[2], dilate=False)
            model.fc = nn.Linear(1024, 100)
        for (name,param) in model.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, nn.init.calculate_gain('relu'))
    
    if params.get("stochastic_depth",0) > 0:
        layers = [3,4,6,3] # From resnet50 definition
        for layer_num in range(4):
            for idx in range(layers[layer_num]):
                seq = getattr(model, f'layer{layer_num+1}') # Get layer
                l = seq.pop(idx) # Remove a bottleneck/block (NOT generalizable to BasicBlock)
                sl = StochasticBottleneck(l, zero_p=params["stochastic_depth"]) # replace with custom bottleneck
                seq.insert(idx, sl) # insert back in nn.Sequential

    return model



class MyImageNet(Dataset):
    def __init__(self, val_file, img_dir, transform=None, target_transform=None, st=None, en=None):
        self.img_labels = pd.read_csv(val_file, sep=' ', header=None, names=['filename', 'class'])
        if st is not None and en is not None:
            self.img_labels = self.img_labels[st:en]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path, mode='RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def imagenet_data_maker(params):
    val_file = '/data/ssd/common/benchmark/datasets/imagenet/val.txt'
    img_dir = '/data/ssd/common/benchmark/datasets/imagenet/val'

    # image_size = params.get('image_size', [224,224])
    fraction = params.get('fraction', 1.0)
    batch_size = params.get('batch_size', 128)
    train_size = max(batch_size, int(40000*fraction))
    test_size = max(batch_size, int(10000*fraction))
    image_size = params.get('image_size', (224,224))

    resize = transforms.Resize([256,256])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # pretrained imagenet
                                         std=[0.229, 0.224, 0.225])
    dtype = transforms.ToDtype(torch.float32,scale=True)

    # transforms_list = [transforms.ToImage(), resize,
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(p=0.5), # Further augmentations can go here.
    #     dtype, normalize
    # ]
    transforms_list = [transforms.ToImage(), 
    transforms.RandomHorizontalFlip(p=0.5),
    resize, transforms.CenterCrop(224)
    ]
    if 'randaugment' in params:
        n,m = params['randaugment']
        transforms_list.append(transforms.RandAugment(num_ops=n,magnitude=int(m*10)))
    if 'randerasing' in params:
        transforms_list.append(transforms.RandomErasing(p=params['randerasing']))
    if 'randomresizedcrop' in params:
        transforms_list.append(transforms.RandomResizedCrop(image_size))
    # else:
    #     transforms_list.append(transforms.RandomCrop(image_size,padding=int(image_size[0]/8)))  # padding with 0s then crop
    if 'autoaugment' in params:
        if params['autoaugment'] == 'cifar10':
            transforms_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        else:
            transforms_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))

    transforms_list.extend([dtype, normalize])
    train_transforms = transforms.Compose(transforms_list)
    test_transforms = transforms.Compose([transforms.ToImage(),
        resize, transforms.CenterCrop(224), dtype, normalize 
    ])
    collate_fn = default_collate

    def both_collate(batch, cutmix_alpha=1.0, mixup_alpha=0.8):
        return transforms.RandomChoice([transforms.MixUp(alpha=mixup_alpha,num_classes=1000), transforms.CutMix(alpha=cutmix_alpha,num_classes=1000)])(*default_collate(batch))

    if 'cutmix_alpha' in params and 'mixup_alpha' in params:
        collate_fn = partial(both_collate, cutmix_alpha=params['cutmix_alpha']
                         , mixup_alpha=params['mixup_alpha'])

    train = MyImageNet(val_file, img_dir,st=0,en=train_size, transform=train_transforms)
    test = MyImageNet(val_file, img_dir, st=train_size,en=(train_size+test_size), transform=test_transforms)
    # train = MyImageNet(val_file, img_dir,st=0,en=4000, transform=train_transforms)
    # test = MyImageNet(val_file, img_dir, st=4000,en=5000, transform=train_transforms)
    
    train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn,
                                   shuffle=True, pin_memory=True, num_workers=16,
                                   prefetch_factor=2, persistent_workers=True, drop_last=True)
    test_dataloader = DataLoader(test, batch_size=batch_size,
                                   shuffle=False, pin_memory=True, num_workers=16, drop_last=True)
    
    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer, device, params=None, epoch=0, log_mflow=True, **kwargs):
    """
        params has hyperparameters (what will be relevant here?)
        Assumes mlflow run has been started.
        Logs average loss, accuracy, periodic batch loss

        if params['distillation'], expects 'teacher' model in kwargs

        Return avg loss, accuracy
    """
    is_distillation = 'distillation' in params and params['distillation']
    # if is_distillation:

        
    model.train()
    optimizer.zero_grad()

    num_batches = len(dataloader)
    batch_log_frequency = max(1,int(num_batches/5)) # logs n'th batch to mlflow

    total_loss = 0 # sum over all batches, loss is already averaged within batch
    accuracy_sum_batches = 0 # sum of accuracy over batches, already averaged within batch
    for (curr_batch, (images, labels)) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        if is_distillation:
            teacher_weight = params['distillation'].get('teacher_weight', 1.0)
            teacher = kwargs['teacher']
            teacher_pred = teacher(images)
            temp = params['distillation'].get('temp', 0.5)
            teacher_loss = teacher_weight*nn.KLDivLoss(reduction='batchmean', log_target=True)(F.log_softmax(teacher_pred/temp, dim=1), F.log_softmax(pred/temp, dim=1))
            if params['distillation'].get('use_ground_labels', True): 
                target_loss = loss_fn(pred, labels)
                loss = target_loss + teacher_loss
                
            else:
                # only use teacher, not label
                if epoch == 0 and curr_batch == 0:
                    print('Using only teacher, not ground labels')
                teacher_labels = teacher_pred.argmax(1)
                target_loss = loss_fn(pred, teacher_labels)
                loss = target_loss + teacher_loss
        else:
            loss = loss_fn(pred, labels)

        total_loss += loss.item()
        # TODO: Revisit following...
        if labels.dim() == 1: # normal test input
            predicted_labels = pred.argmax(1)
            accuracy_sum_batches += (predicted_labels == labels).mean(dtype=torch.float32).item()
        else: # mixed labels (cutmix etc). NOTE: Just do dominant class for now, but this is not ideal
            predicted_labels = pred.argmax(1)
            dominant_labels = labels.argmax(1)
            accuracy_sum_batches += (predicted_labels == dominant_labels).mean(dtype=torch.float32).item()
        
        loss.backward()
        optimizer.step()
        if 'scheduler' in kwargs and (isinstance(kwargs['scheduler'], OneCycleLR)):
            kwargs['scheduler'].step() # step every batch for OneCycleLR scheduler
        optimizer.zero_grad()

        if curr_batch % batch_log_frequency == 0 and log_mflow:
            mlflow.log_metric('batch_loss', loss.item(), step=curr_batch+epoch*num_batches)
            if is_distillation:
                mlflow.log_metric('teacher_loss_scaled', teacher_loss.item(), step=curr_batch+epoch*num_batches)
                mlflow.log_metric('target_loss', target_loss.item(), step=curr_batch+epoch*num_batches)
    
    avg_loss = total_loss/num_batches
    accuracy = accuracy_sum_batches/num_batches
    if log_mflow:
        mlflow.log_metrics({
            'training_loss': avg_loss,
            'training_accuracy': accuracy
        }, step=epoch)

    return (avg_loss, accuracy)

def test(dataloader, model, loss_fn, device, epoch=0, log_mlflow=True):
    model.eval()

    num_batches = len(dataloader)
    total_loss = 0 # see train
    accuracy_sum_batches = 0 
    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = loss_fn(pred, labels)

            if labels.dim() == 1: # normal test input
                predicted_labels = pred.argmax(1)
                accuracy_sum_batches += (predicted_labels == labels).mean(dtype=torch.float32).item()
            else: # mixed labels (cutmix etc). NOTE: Just do dominant class for now, but this is not ideal
                predicted_labels = pred.argmax(1)
                dominant_labels = labels.argmax(1)
                accuracy_sum_batches += (predicted_labels == dominant_labels).mean(dtype=torch.float32).item()
            
            total_loss += loss.item()

    avg_loss = total_loss/num_batches
    accuracy = accuracy_sum_batches/num_batches
    if log_mlflow:
        mlflow.log_metrics({
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }, step=epoch)

    return (avg_loss, accuracy)
    
def train_epochs(train_dl, test_dl, model, loss_fn, optimizer, scheduler, device, params, total_epochs=5, log_mlflow=True, epoch_callback=None, **kwargs):
    """
    if params['distillation'], expects 'teacher' model in kwargs (passed it to train)
    """
    test_acc = None
    best_test_acc = -1
    best_model = None

    if log_mlflow:
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            name = params['name']

            if total_epochs > 0:
                test_loss, test_acc = test(test_dl, model, loss_fn, device=device, epoch=0, log_mlflow=log_mlflow)
                print(f'Epoch 0: Test Acc.: {test_acc*100.0:.2f}%')
                mlflow.log_metrics({
                            'test_loss': test_loss,
                            'test_accuracy': test_acc
                        })

            if scheduler and ( isinstance(scheduler, OneCycleLR)):
                kwargs['scheduler'] = scheduler

            for epoch in range(total_epochs):
                if 'training' in params:
                    # Custom training stuff here, e.g.freeze some layers.
                    if params['training']['type'] == 'fine-tune':
                        # expects params['training']['threshold'] = some epoch
                        # Until then, train only fc layer
                        if epoch < params['training']['threshold']:
                            for parameter in model.parameters():
                                parameter.requires_grad = False
                            for parameter in model.fc.parameters():
                                parameter.requires_grad = True
                        else:
                            # Train all
                            for (name, parameter) in model.named_parameters():
                                parameter.requires_grad = True
                
                train_loss, train_acc = train(train_dl, model, loss_fn, optimizer, device=device, params=params, epoch=epoch, log_mflow=log_mlflow, **kwargs)
                # print(train_loss, train_acc)
                test_loss, test_acc = test(test_dl, model, loss_fn, device=device, epoch=epoch, log_mlflow=log_mlflow)
                # print(test_loss, test_acc)
                

                if scheduler:
                    if not isinstance(scheduler, OneCycleLR):
                        scheduler.step()
                    mlflow.log_metric('current_lr', scheduler.get_last_lr()[0], step=epoch)

                if test_acc > best_test_acc:
                    best_model = model
                    torch.save(best_model.state_dict(), f"saved_models/{name}-best")
                    best_test_acc = test_acc

                if params['trial']:
                    params['trial'].report(test_acc, epoch)

                    if params['trial'].should_prune():
                        raise optuna.exceptions.TrialPruned()
                torch.save(model.state_dict(), f"saved_models/{name}-checkpoint")
                if epoch_callback:
                    epoch_callback(epoch, model, train_acc, test_acc, params)
                    
            if total_epochs > 0:
                mlflow.log_metrics({
                            'training_loss': train_loss,
                            'training_accuracy': train_acc,
                            'test_loss': test_loss,
                            'test_accuracy': test_acc
                        })
            
            torch.save(model.state_dict(), f"saved_models/{name}-final")
    else:   
        for epoch in range(total_epochs):
            print(f"Epoch {epoch+1}/{total_epochs}..")

            train_loss, train_acc = train(train_dl, model, loss_fn, optimizer, device=device, params=params, epoch=epoch, log_mflow=log_mlflow)
            test_loss, test_acc = test(test_dl, model, loss_fn, device=device, epoch=epoch, log_mlflow=log_mlflow)

            if params['trial']:
                params['trial'].report(test_acc, epoch)

                if params['trial'].should_prune():
                    raise optuna.exceptions.TrialPruned()

            torch.save(model.state_dict(), f"saved_models/checkpoint")
        torch.save(model.state_dict(), f"saved_models/final")
        
    if test_acc is not None:
        return test_acc
    return None


def my_train_scheme_maker(params, model):
    if 'loss_fn' not in params:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0))
    else:
        if params['loss_fn'] == 'ce':
            loss_fn = nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0))
        elif params['loss_fn'] == 'bce':
            loss_fn = nn.BCEWithLogitsLoss()


    optimizer_type = params.get('optimizer_type', 'SGD')
    params.setdefault('weight_decay',0)
    params.setdefault('lr',1e-2)
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], 
                                    momentum=params.get('momentum',0.9), 
                                    weight_decay=params['weight_decay'], nesterov=True)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'])


    if 'lr_schedule' in params:
        if params['lr_schedule']['type'] == 'step':
            warmup_schedule =  LinearLR(optimizer, start_factor=1e-4,total_iters=params['lr_schedule']['warmup'])
            rest_schedule = StepLR(optimizer, step_size=params['lr_schedule']['epochs'], gamma=params['lr_schedule']['rate'])
            scheduler = ChainedScheduler([warmup_schedule, rest_schedule], optimizer=optimizer) 
            # Note, this is not exactly sequential but both schedules 'stack'
        elif params['lr_schedule']['type'] == 'cosine':
            warmup_schedule =  LinearLR(optimizer, start_factor=1e-4,total_iters=params['lr_schedule']['warmup'])
            rest_schedule = CosineAnnealingLR(optimizer,params['total_epochs'])
            scheduler = ChainedScheduler([warmup_schedule, rest_schedule], optimizer=optimizer) 
        elif params['lr_schedule']['type'] == '1cycle':
            scheduler = OneCycleLR(optimizer, max_lr=params['lr'], epochs=params['total_epochs'], steps_per_epoch=params['steps_per_epoch'])
    else: 
        scheduler = None

    return optimizer, loss_fn, scheduler
