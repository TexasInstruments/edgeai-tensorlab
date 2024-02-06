from edgeai_torchmodelopt.xmodelopt.surgery.v2 import convert_to_lite_fx
from edgeai_torchmodelopt.xmodelopt.pruning import PrunerModule
from edgeai_torchmodelopt.xmodelopt.quantization.v2 import QATFxModule
from ptflops import get_model_complexity_info

import torch
from torchvision import datasets, transforms, models
from torchinfo import s
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import onnx, onnxsim

import datetime as dt
import copy

n_epochs = 6
sp_epochs = 10
batch_size_train = 512
batch_size_test = 512
learning_rate = 0.1
momentum = 0.5
log_interval = 10
save_ckpt = True


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

_str_to_layer_dict ={
    'ReLU':nn.ReLU(),
    'ReLU6':nn.ReLU6(),
    'GELU':nn.GELU(),
}


def _str_to_layer(surgery_dict:dict):
    repl_dict ={}
    for key,val in surgery_dict.items():
        repl_dict.update({_str_to_layer_dict[key]:_str_to_layer_dict[val]})
    return repl_dict


def export_model(model:nn.Module,dummyInput:torch.Tensor,onnxFileName:str,simp_flag:bool=False):
    if isinstance(model,nn.DataParallel):
        model=model.module
    fileDescr= onnxFileName.rsplit('.',1)
    onnxFileName=fileDescr[0]
    intrmdtfileName1=str(onnxFileName)+'.onnx'
    # torch.save(model.state_dict(),str(onnxFileName)+'   .ckpt')
    torch.onnx.export(model,dummyInput,intrmdtfileName1)#, training=torch.onnx.TrainingMode.TRAINING)
    if simp_flag:
        loadedModel = onnx.load(intrmdtfileName1)
        simplifiedModel,check= onnxsim.simplify(loadedModel)
        assert check,'Simpplification Failed'
        onnx.save(simplifiedModel,str(onnxFileName)+'.onnx')
        # torch.save(simplifiedModel.mo,str(onnxFileName)+'_simplified.ckpt')
    return simplifiedModel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)
        self.act1 = nn.ReLU6() 
        self.act2 = nn.ReLU() 
        self.act3 = nn.GELU() 

    def forward(self, x):
        x = self.act1(F.max_pool2d(self.conv1(x), 2))
        x = self.act2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = self.act3(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



model_dict = {
    'mobilenet_v2':models.mobilenet_v2,
    'efficientnet_b0':models.efficientnet_b0,
    'convnext_tiny':models.convnext_tiny,
    'regnet_y_8gf':models.regnet_y_8gf,
}

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': batch_size_train}
test_kwargs = {'batch_size': batch_size_test}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.CenterCrop(160)
    ])
# dataset1 = datasets.CIFAR10('../data', train=True, download=True,
#                    transform=transform)
# dataset2 = datasets.CIFAR10('../data', train=False,
#                    transform=transform)
path = 'imagenette'
try:
    dataset1 = datasets.ImageFolder(f"{path}/train",transform=transform)
    dataset2 = datasets.ImageFolder(f"{path}/val",transform=transform)
except:
    path = 'edgeai-modelopt_demo/' +path
    dataset1 = datasets.ImageFolder(f"{path}/train",transform=transform)
    dataset2 = datasets.ImageFolder(f"{path}/val",transform=transform)

train_loader = DataLoader(dataset1,  **train_kwargs)
test_loader = DataLoader(dataset2, **test_kwargs)

def backend_task(model_type="regnet_y_8gf",n_epochs=n_epochs,surgery_dict=None,sp_ratio=0.0,sp_type='channel',qntzn=False,save_ckpt = False):
    train_losses = []
    test_losses = []
    
    if surgery_dict is None:
        surgery_dict = {}
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    
    model_type = model_dict[model_type]
    model = model_type(num_classes=10).to(device)
    model_without_ddp = model
    checkpoint = torch.load('checkpoint.pth',   map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint["model"],strict=False)
    
    if surgery_dict and len(surgery_dict)>0:
        print('before surgery:')
        print(model)
        # export_model(model,torch.rand(1,1,28,28),'beforesg.onnx',True)
        surgery_dict=_str_to_layer(surgery_dict)
        # model = copy.deepcopy(model)
        model = convert_to_lite_fx(model, surgery_dict)
        # export_model(model,torch.rand(1,1,28,28),'aftersg.onnx',True)
        print('after surgery:')
        print(model)
    else:
        # model = torch.fx.symbolic_trace(model)
        print("Network")
        
    # model = torch.nn.DataParallel(model)
    # model_without_ddp = model.module
    # print(model)    
    
    def train(model,epoch,optimizer,criterion,device):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            # torch.save(model.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
    
    def test(model,criterion, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        accuracy=100. * correct / len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),accuracy
            ))
        return accuracy

    criterion = nn.CrossEntropyLoss()
    def train_model(model,no_of_epochs,device):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_acc = test(model,criterion,device)
        for epoch in range(1, no_of_epochs + 1):
            train(model,epoch,optimizer,criterion,device)
            acc = test(model,criterion,device)
            if save_ckpt and best_acc<acc:
                checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                }
                best_acc = acc
                torch.save(checkpoint,  "checkpoint.pth")
                
              
        return acc
    
    t1 = dt.datetime.now()
    acc1 =test(model,criterion,device)
    # print(type(model))
    t2 = dt.datetime.now()
    print((t2-t1).total_seconds(),'seconds')
    flop_count1,params1 = get_model_complexity_info(model,tuple(torch.Tensor(dataset2[0][0]).shape),as_strings=False,print_per_layer_stat=False)
    acc = None
    params = None
    flop_count = None
    
    if sp_ratio>0 and qntzn:
        print("Both Sparsity and Quantization are not supported simultaneously.")
    else:
        # export_model(model,torch.rand(1,1,28,28),'beforesp.onnx',True)
        sp_rto_done = None
        if sp_ratio >0:
            if surgery_dict and len(surgery_dict)>0:
                sg_epochs = n_epochs
            else:
                sg_epochs =0
            print(f"Apllying Sparsity with ratio {sp_ratio} and type {sp_type} for {sp_epochs} epochs out of {(sp_epochs+sg_epochs)}")
            model = PrunerModule(model,pruning_ratio=sp_ratio,total_epochs=(sg_epochs+sp_epochs),pruning_init_train_ep=sg_epochs,pruning_type=sp_type)
            model=torch.nn.DataParallel(model)
            acc = train_model(model,sp_epochs,device)
            sp_rto_done= model.module.sparsity if isinstance(model.module,PrunerModule) else 0 
        if qntzn:
            if surgery_dict and len(surgery_dict)>0 and sp_ratio ==0:
                model = nn.DataParallel(model)
                acc = train_model(model,n_epochs,device)
            print(f"Apllying Quantization for {sp_epochs} epochs")
            if isinstance(model,torch.nn.DataParallel):
                model= QATFxModule(model.module,total_epochs =sp_epochs)
            else:
                model= QATFxModule(model,total_epochs =sp_epochs)
            model=torch.nn.DataParallel(model)
            acc = train_model(model,sp_epochs,device)
        if sp_ratio == 0 and not qntzn :
            model = nn.DataParallel(model)
            acc = train_model(model,n_epochs,device)
            # export_model(model,torch.rand(1,1,28,28),'aftersp.onnx',True)
        
        flop_count,params = get_model_complexity_info(model.module if isinstance(model,nn.DataParallel) else model,tuple(torch.Tensor(dataset2[0][0]).shape),as_strings=False,print_per_layer_stat=True)
        print(flop_count,params) 
        if sp_rto_done :
            params = int(params * (1-(model.module.sparsity )) )
            flop_count = int(flop_count*(1-model.module.sparsity)) 
            print(flop_count,params) 
    
    print(f'''
            Sparsity Ratio: {sp_ratio}
            Sparsity Type: {sp_type}
            Quantization: {qntzn}
            Accuracy: {acc1} -> {acc}
            Flop Count: {flop_count1} -> {flop_count or "None"}
            Number of parameters: {params1} -> {params or params1}
          ''')
    return acc or acc1,flop_count or flop_count1


if __name__ == '__main__':    
    backend_task(list(model_dict.keys())[0],n_epochs=25 ,save_ckpt=save_ckpt) 
    # backend_task({'relu':'gelu'})
    # backend_task({'GELU':'ReLU','ReLU6':'ReLU'})
    