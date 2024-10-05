#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

#package imports
import torch, onnx, onnxsim
import torch.nn as nn
import wandb
import os
try:
    import torchvision
    has_tv = True
except:
    has_tv = False
if has_tv:
    from torchvision import datasets, transforms
from torch import distributed as dist
from enum import Enum
import time


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def __init__():

    torch.autograd.set_detect_anomaly(True)
def _initializeWandb(projectName:str,lr=0.001,arch=None,datadir:str=None,epochs=100):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=projectName,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": arch,
        "dataset":datadir,
        "epochs":epochs,
        }
    )

def _initializeDevice():
    return  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if has_tv:
    def _initializeDataLoader(dataDir:str,batchSize:int=256):
        traindir = os.path.join(dataDir, 'train')
        valdir = os.path.join(dataDir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batchSize, shuffle=True, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batchSize, shuffle=False, pin_memory=True )
        return train_loader,val_loader


def exportAndSimplifyOnnx(model:nn.Module,dummyInput:torch.Tensor,onnxFileName:str):
    if isinstance(model,nn.DataParallel):
        model=model.module
    fileDescr= onnxFileName.rsplit('.',1)
    onnxFileName=fileDescr[0]
    intrmdtfileName1=str(onnxFileName)+'.onnx'
    torch.save(model.state_dict(),str(onnxFileName)+'   .ckpt')
    torch.onnx.export(model,dummyInput,intrmdtfileName1)#, training=torch.onnx.TrainingMode.TRAINING)
    loadedModel = onnx.load(intrmdtfileName1)
    simplifiedModel,check= onnxsim.simplify(loadedModel)
    assert check,'Simpplification Failed'
    onnx.save(simplifiedModel,str(onnxFileName)+'_simplified.onnx')
    # torch.save(simplifiedModel.mo,str(onnxFileName)+'_simplified.ckpt')
    return simplifiedModel

def trainModel(model:nn.Module,dataDir:str,projectName:str='',epochs=100,lr=0.001,
               criterion=None,optimizer=None,scheduler=None):
    model.train()
    device=_initializeDevice()
    model=model.to(device)
    model=nn.DataParallel(model,device_ids=[0,1,2,3])
    if criterion==None:
        criterion=nn.CrossEntropyLoss().to(device)
    # if optimizer==None:s
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    # if scheduler==None:
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)
    trainLoader,valLoader= _initializeDataLoader(dataDir)
    if projectName == '':
        print(  '''
        Project Name is none.
        So, This run will be for checking whether model is running  or not.
        ''')
        pass
    else:
        _initializeWandb(projectName,lr,type(model).__name__,dataDir,epochs)
        print('wandb started')
    best_acc1=0
    for epoch in range(epochs):
        model.train()

        # train for one epoch
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainLoader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        # switch to train mode
        end = time.time()
        for i, (images, target) in enumerate(trainLoader):
            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i + 1)
        if projectName!='':
            wandb.log({"top_acc1": top1.avg, "top_acc5": top5.avg,"batch_time":batch_time.avg ,"data_time":data_time.avg ,"loss": losses.avg})

        # evaluate on validation set
        acc1 = validate(valLoader, model, criterion,device)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

    if projectName!='':
        wandb.finish()
    # torch.save(model.state_dict(), projectName+'.ckpt')
    
    
def validate(val_loader, model, criterion,device):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + 0,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg
