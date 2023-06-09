import torch
from torch import nn,fx,Tensor
from torch.fx import symbolic_trace,GraphModule, Node
from typing import Dict, Any, Union
import operator
from copy import deepcopy

# moduleReplacementdict=Dict[]()
# functionReplacementDict=Dict[]()
def _get_parent_name(target:str):
    *parent, name = target.rsplit('.', 1)
    return ( parent[0] if parent else ''), name

def replaceModuleNode(node:Node,modulesDict:Dict[str,Any],replaceModel:Union[GraphModule,nn.Module]):
    if node.op != 'call_module':
        print('''
        Not a module Node!
        So, No changes will be made.
        ''')
        return 
    parent_name, name = _get_parent_name(node.target)
    modulesDict.update(node.target,replaceModel)
    setattr(modulesDict[parent_name], name, replaceModel)

def replaceFunctionNode(node:Node,tracedModel:GraphModule,replaceFunction):
    if node.op !='call_function':
        print('''
        Not a function Node!
        So, No changes will be made.
        ''')
        return     
    with tracedModel .graph.inserting_after(node):
        new_node = tracedModel.graph.call_function(replaceFunction, node.args, node.kwargs)
        node.replace_all_uses_with(new_node)
    # Remove the old node from the graph
    tracedModel.graph.erase_node(node)

def _areBothNodeEqual(firstNode:Node,secondNode:Node,firstGModule:Union[GraphModule,None]=None,secondGModule:Union[GraphModule,None]=None):
    operationDict={torch.add:operator.add,torch.sub:operator.sub,torch.mul:operator.mul,operator.add:torch.add,operator.sub:torch.sub,operator.mul:torch.mul}
    if firstNode.op==secondNode.op:
        if firstNode.op == 'placeholder':
            return len(firstNode.users)==len(secondNode.users)
        if firstNode.op=='output':
            return (len(firstNode.args)+len(firstNode.kwargs))==(len(secondNode.args)+len(secondNode.kwargs))
        if firstNode.op =='call_function':
            if firstNode.target==secondNode.target:
                return True
            elif firstNode.target in operationDict.keys():  
                return secondNode.target == operationDict[firstNode.target]
            else: return False
        if firstNode.op =='call_method':
            return firstNode.target==secondNode.target
        if (firstGModule==None) or (secondGModule==None):
            print("\nGraphModules are required for both nodes\nas at least one of them is 'call_module' node.")
            return None
        modulesIn1stGraph = dict(firstGModule.named_modules())
        modulesIn2ndGraph = dict(secondGModule.named_modules())
        try:
            module1=modulesIn1stGraph[firstNode.target]
            module2=modulesIn2ndGraph[secondNode.target]
        except:
            print("Exception",firstGModule.get_submodule('features.4.block.0.2'))
            print(modulesIn1stGraph['features.4.block.0.2'])
        return str(type(module1))== str(type(module2))
    
    return False


def _straightChainSearcher(mainModule:GraphModule,patternModule:GraphModule):
    mainModuleNodes:list[fx.Node] =list(mainModule.graph.nodes)
    patternModuleNodes:list[fx.Node] =list(patternModule.graph.nodes)
    count={'placeholder':0,'output':0}
    for patternNode in patternModuleNodes:
        if (patternNode.op in ['placeholder','output']):
            patternModuleNodes.remove(patternNode)
            if patternNode.op=='placeholder':
                count[patternNode.op] +=1
            elif patternNode.op=='output':
                count[patternNode.op] = len(patternNode.args)
    assert count['output']==1 and count['placeholder']==1, 'This function is not for multi-input or multi-output'
    mainModuleNodeNum=len(mainModuleNodes)
    patternModuleNodeNum=len(patternModuleNodes)

    mainIndex=0
    pattIndex=0
    matched=list()
    secondStartIndex=-1
    inp,out=-1,-1
    while(mainIndex<mainModuleNodeNum):
        mainNode=mainModuleNodes[mainIndex]
        patnNode=patternModuleNodes[pattIndex]
        cond=_areBothNodeEqual(mainNode,patnNode,mainModule,patternModule)
        if (cond):
            if mainNode==patternModuleNodes[0] and secondStartIndex==-1 and pattIndex!=0:
                    secondStartIndex=mainIndex
            if pattIndex==0:
                inp=mainNode
            if pattIndex==(patternModuleNodeNum-1):
                out=mainNode
                matched.append((inp,out))
                secondStartIndex=-1
            pattIndex+=1
            pattIndex=pattIndex%patternModuleNodeNum
            mainIndex+=1
        else:
            inp,out=-1,-1
            pattIndex=0
            if secondStartIndex==-1:
                mainIndex+=1
            else:
                mainIndex=secondStartIndex
                secondStartIndex=-1
    return matched

def _replacePattern(mainModule:GraphModule,start:Node,end:Node,replaceModule:GraphModule,no_of_module_replaced:int=0):
    new_node_name='replaced_'+str(replaceModule.__class__.__name__)+'_'+str(no_of_module_replaced)
    mainModules=dict(mainModule.named_modules())
    mainModule.add_module(new_node_name,replaceModule)
    replaceModule=deepcopy(replaceModule)
    with mainModule.graph.inserting_before(start):
        newNode = mainModule.graph.call_module(new_node_name,start.args,start.kwargs)
        ptr=start
        while ptr!=end:
            if (ptr.op=='call_module'):
                parent_name,name= _get_parent_name(ptr.target)
                mainModules[parent_name].__delattr__(name)
            ptr.replace_all_uses_with(newNode)
            temp=ptr.next
            mainModule.graph.erase_node(ptr)
            ptr=temp
        if (ptr.op=='call_module'):
            parent_name,name= _get_parent_name(end.target)
            mainModules[parent_name].__delattr__(name)
        end.replace_all_uses_with(newNode)
    mainModule.graph.erase_node(end)
    # print(mainModules)
    mainModules.update({new_node_name:replaceModule})
    # print(mainModules)
    no_of_module_replaced+=1
    pass
def _replacePattern1(mainModule:GraphModule,start:Node,end:Node,replaceModule:GraphModule,no_of_module_replaced:int=0):
    replaceModule= deepcopy(replaceModule)
    mainModules=dict(mainModule.named_modules())
    if (start.op=='call_module'):
        parent_name,name =_get_parent_name(start.target)
        parent_module=mainModules[parent_name]
        parent_module.__setattr__(name,replaceModule)
        mainModules[start.target]=replaceModule
        newNode=start
        ptr=start.next
        if start!=end:
            while ptr!=end:
                if (ptr.op=='call_module'):
                    parent_name,name= _get_parent_name(ptr.target)
                    parent_module.__delattr__(name)
                ptr.replace_all_uses_with(newNode)
                temp=ptr.next
                mainModule.graph.erase_node(ptr)
                ptr=temp
            if (ptr.op=='call_module'):
                parent_name,name= _get_parent_name(end.target)
                parent_module.__delattr__(name)
        # print(parent_module.graph.print_tabular())
    else:
        new_node_name='replaced_'+str(replaceModule.__class__.__name__)+'_'+str(no_of_module_replaced)
        mainModule.add_module(new_node_name,replaceModule)
        with mainModule.graph.inserting_before(start):
            newNode = mainModule.graph.call_module(new_node_name,start.args,start.kwargs)
            ptr=start
            while ptr!=end:
                if (ptr.op=='call_module'):
                    parent_name,name= _get_parent_name(ptr.target)
                    mainModules[parent_name].__delattr__(name)
                ptr.replace_all_uses_with(newNode)
                temp=ptr.next
                mainModule.graph.erase_node(ptr)
                ptr=temp
            if (ptr.op=='call_module'):
                parent_name,name= _get_parent_name(end.target)
                mainModules[parent_name].__delattr__(name)
            end.replace_all_uses_with(newNode)
        mainModule.graph.erase_node(end)
        mainModules.update({new_node_name:replaceModule})
    # print(mainModules)
    # print(mainModules)
    mainModule.recompile()
    pass

def graphPatternReplacer(mainModule:GraphModule,patternModule:Union[GraphModule,nn.Module,callable],replaceModule:Union[GraphModule,nn.Module,callable]):
    if type(patternModule)!=GraphModule:
        patternModule=symbolic_trace(patternModule)
    if type(replaceModule)!=GraphModule:
        replaceModule=symbolic_trace(replaceModule)
    global no_of_module_replaced
    no_of_module_replaced=0
    # print(patternModule)
    # print(replaceModule)
    patternNodes=[]
    numberOfInput=0
    numberOfOutput=0
    for node in patternModule.graph.nodes:
        if node.op=='placeholder':
            numberOfInput+=1
        elif node.op =='output':
            numberOfOutput=len(node.args)
        else:
            patternNodes.append(node)
    #singleNode
    # if (len(patternNodes)==1):
    #     # (patternNodes[0].op)
    #     if patternNodes[0].op=='call_module':
    #         for node in mainModule.graph.nodes:
    #             if _areBothNodeEqual(node,patternNodes[0],mainModule,patternModule):
    #                _replacePattern(mainModule,node,node,replaceModule)
    #     elif patternNodes[0].op=='call_function':
    #         for node in mainModule.graph.nodes:
    #             cond=_areBothNodeEqual(node,patternNodes[0],mainModule,patternModule)
    #             if cond:
    #                 mainModule.add_module('replaced'+str(replaceModule.__class__.__name__),replaceModule)
    #                 with mainModule .graph.inserting_before(node):
    #                     new_node = mainModule.graph.call_module(replaceModule, node.args, node.kwargs)
    #                     node.replace_all_uses_with(new_node)
    #                 mainModule.graph.erase_node(node)
    #                 # Remove the old node from the graph
    #                 # TODO may delete node later on
    #                 # mainModule.graph.erase_node(node)
    #     elif patternNodes[0].op=='call_method':
    #         for node in mainModule.graph.nodes:
    #             if _areBothNodeEqual(node,patternNodes[0],mainModule,patternModule):
    #                 mainModule.add_module('replaced'+str(replaceModule.__class__.__name__),replaceModule)
    #                 with mainModule .graph.inserting_before(node):
    #                     new_node = mainModule.graph.call_module(replaceModule, node.args, node.kwargs)
    #                     node.replace_all_uses_with(new_node)
    #                 # TODO may delete node later on
    #         #one input one output
    # el
    if numberOfInput ==1 and numberOfOutput==1:
        matches = _straightChainSearcher(mainModule,patternModule)
        for (start,end) in matches:
            _replacePattern1(mainModule,start,end,replaceModule,no_of_module_replaced)
            no_of_module_replaced+=1
    # delete_unused_nodes(mainModule)
    mainModule.graph.lint()
    print(no_of_module_replaced)

    return mainModule


def delete_unused_nodes(graph_module:GraphModule):
    nodes=list()
    for node in graph_module.graph.nodes:
        nodes.append(node)
    n=len(nodes)
    currNode=nodes[-1]
    while currNode:
        temp=currNode
        currNode=currNode.prev
        if temp.op== 'output':
            continue
        if len(temp.users.keys())==0:
            graph_module.graph.erase_node(temp)

from copy import deepcopy


def replaceAndExpot(mainModule:nn.Module,dummyinput:Tensor,patternModule:Union[GraphModule,nn.Module,callable],replaceModule:Union[GraphModule,nn.Module,callable]):
    mainModule =deepcopy(mainModule)
    mainModule=symbolic_trace(mainModule)
    exportAndSimplifyOnnx(mainModule,dummyinput,str(type(mainModule))+'before.onnx')
    resultModel=graphPatternReplacer(mainModule,patternModule,replaceModule)
    exportAndSimplifyOnnx(mainModule,dummyinput,str(type(mainModule))+'after.onnx')
    return resultModel