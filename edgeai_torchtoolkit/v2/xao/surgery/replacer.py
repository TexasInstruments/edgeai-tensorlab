import torch
from inspect import getmodule, isfunction
from torch import nn,fx,Tensor
from torch.fx import symbolic_trace,GraphModule, Node
from typing import Dict, Any, Union, List
import operator
from copy import deepcopy
from .custom_module import *


__all__ = ['replace_unsuppoted_layers', 'get_replacement_dict_default']


# moduleReplacementdict=Dict[]()
# functionReplacementDict=Dict[]()
def _get_parent_name(target:str):
    *parent, name = target.rsplit('.', 1)
    return ( parent[0] if parent else ''), name

def replace_module_node(node:Node,modules_dict:Dict[str,Any],replace_model:Union[GraphModule,nn.Module]):
    if node.op != 'call_module':
        print('''
        Not a module Node!
        So, No changes will be made.
        ''')
        return 
    parent_name, name = _get_parent_name(node.target)
    modules_dict.update(node.target,replace_model)
    setattr(modules_dict[parent_name], name, replace_model)

def replace_function_node(node:Node,traced_model:GraphModule,replace_function):
    if node.op !='call_function':
        print('''
        Not a function Node!
        So, No changes will be made.
        ''')
        return     
    with traced_model .graph.inserting_after(node):
        new_node = traced_model.graph.call_function(replace_function, node.args, node.kwargs)
        node.replace_all_uses_with(new_node)
    # Remove the old node from the graph
    traced_model.graph.erase_node(node)

#checks whether two nodes are  equal or not
def _are_both_node_equal(first_node:Node,second_node:Node,first_graoh_module:Union[GraphModule,None]=None,second_graph_module:Union[GraphModule,None]=None):
    # till now for two node to be same they must have same operation
    #for operator and torch function counter-parts
    operationDict={torch.add:operator.add,torch.sub:operator.sub,torch.mul:operator.mul,operator.add:torch.add,operator.sub:torch.sub,operator.mul:torch.mul}
    if first_node.op==second_node.op:
        if first_node.op == 'placeholder':
            # for placeholder both should have same number of users
            return len(first_node.users)==len(second_node.users)
        if first_node.op=='output':
            # for output both should have same number of argument i.e. number of outputs are same 
            return (len(first_node.args)+len(first_node.kwargs))==(len(second_node.args)+len(second_node.kwargs))
        if first_node.op =='call_function':
            if first_node.target==second_node.target:
                #if both refer to same function
                return True
            elif first_node.target in operationDict.keys():  
                #if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
                return second_node.target == operationDict[first_node.target]
            else: return False
        if first_node.op =='call_method':
            #both should refer to same method
            return first_node.target==second_node.target
        if (first_graoh_module==None) or (second_graph_module==None):
            print("\nGraphModules are required for both nodes\nas at least one of them is 'call_module' node.")
            return None
        modules_in_1st_graph = dict(first_graoh_module.named_modules())
        modules_in_2nd_graph = dict(second_graph_module.named_modules())
        module_1=modules_in_1st_graph[first_node.target]
        module_2=modules_in_2nd_graph[second_node.target]
        #TODO change equality module fpr hyperparameters
        #both should be objects of same class
        return str(type(module_1))== str(type(module_2))
    
    return False

#searches pattern with on single input and one output other wise a single node with out checking kwargs
def straight_chain_searcher(main_module:GraphModule,pattern_module:GraphModule):
    main_module_nodes:list[fx.Node] =list(main_module.graph.nodes)
    pattern_module_nodes=[]
    number_of_input=0
    number_of_output=0
    for node in pattern_module.graph.nodes:
        if node.op=='placeholder':
            number_of_input+=1
        elif node.op =='output':
            number_of_output=len(node.args)
        else:
            pattern_module_nodes.append(node)
    main_module_node_num=len(main_module_nodes)
    pattern_module_node_num=len(pattern_module_nodes)
    assert (number_of_input==1 and number_of_output==1) or pattern_module_node_num==1, 'This function is not for multi-input or multi-output'

    main_index=0
    patt_index=0
    matched=list()
    second_start_index=-1
    inp,out=-1,-1
    while(main_index<main_module_node_num):
        main_node=main_module_nodes[main_index]
        patn_node=pattern_module_nodes[patt_index]
        cond=_are_both_node_equal(main_node,patn_node,main_module,pattern_module)
        if (cond):
            if main_node==pattern_module_nodes[0] and second_start_index==-1 and patt_index!=0:
                    second_start_index=main_index
            if patt_index==0:
                inp=main_node
            if patt_index==(pattern_module_node_num-1):
                out=main_node
                matched.append((inp,out))
                second_start_index=-1
            patt_index+=1
            patt_index=patt_index%pattern_module_node_num
            main_index+=1
        else:
            inp,out=-1,-1
            patt_index=0
            if second_start_index==-1:
                main_index+=1
            else:
                main_index=second_start_index
                second_start_index=-1
    return matched

#replaces a pattern fom start to end in graph module to a call_module node
def _replace_pattern(main_module:GraphModule,start:Node,end:Node,replace_module:nn.Module,no_of_module_replaced:int=0):
    main_modules=dict(main_module.named_modules())
    if start == end:
            if start in ['call_function','call_method']:
                traced_replacement=symbolic_trace(replace_module)
                replcament_nodes=[]
                for node in traced_replacement.graph.nodes:
                    if node.op not in ['placeholder','output']:
                        replcament_nodes.append(node)
                if len(replcament_nodes) == 1:
                    if  replcament_nodes[0].op == 'call_function':
                        with main_module.graph.inserting_after(start):
                            new_node= main_module.graph.call_function(replcament_nodes[0].target,start.args,start.kwargs)
                            start.replace_all_uses_with(new_node)
                        print(start)
                        main_module.graph.erase_node(start)
                        main_module.recompile()
                        return
                    elif replcament_nodes[0].op == 'call_method':
                        with main_module.graph.inserting_after(start):
                            new_node= main_module.graph.call_method(replcament_nodes[0].target,start.args,start.kwargs)
                            start.replace_all_uses_with(new_node)
                        main_module.graph.erase_node(start)
                        main_module.recompile()
                        return
                        
    if (start.op=='call_module'):
        parent_name,name =_get_parent_name(start.target)
        parent_module=main_modules[parent_name]
        parent_module.__setattr__(name,replace_module)
        main_modules[start.target]=replace_module
        newNode=start
        ptr=start.next
        if start!=end:
            while ptr!=end:
                if (ptr.op=='call_module'):
                    parent_name,name= _get_parent_name(ptr.target)
                    parent_module.__delattr__(name)
                ptr.replace_all_uses_with(newNode)
                temp=ptr.next  
                main_module.graph.erase_node(ptr)
                ptr=temp
            if (end.op=='call_module'):
                parent_name,name= _get_parent_name(end.target)
                parent_module.__delattr__(name)
            end.replace_all_uses_with(newNode)
            main_module.graph.erase_node(end)
        # print(parent_module.graph.print_tabular())
    else:
        new_node_name='replaced_'+str(replace_module.__class__.__name__)+'_'+str(no_of_module_replaced)
        main_module.add_module(new_node_name,replace_module)
        with main_module.graph.inserting_before(start):
            newNode = main_module.graph.call_module(new_node_name,start.args,start.kwargs)
            ptr=start
            while ptr!=end:
                if (ptr.op=='call_module'):
                    parent_name,name= _get_parent_name(ptr.target)
                    main_modules[parent_name].__delattr__(name)
                ptr.replace_all_uses_with(newNode)
                temp=ptr.next
                main_module.graph.erase_node(ptr)
                ptr=temp
            if (ptr.op=='call_module'):
                parent_name,name= _get_parent_name(end.target)
                main_modules[parent_name].__delattr__(name)
            end.replace_all_uses_with(newNode)
            main_module.graph.erase_node(end)
        main_modules.update({new_node_name:replace_module})
    # print(main_modules)
    # print(main_modules)
    main_module.graph.lint()
    main_module.recompile()

#replaces all matches with call_module node
def _replace_all_matches(main_module:GraphModule,matches,replace_module:nn.Module):
    no_of_module_replaced=0
    for (start,end) in matches:
        replace_module_copy =deepcopy(replace_module)
        _replace_pattern(main_module,start,end,replace_module_copy,no_of_module_replaced)
        no_of_module_replaced+=1

#replace nodes if they don't need any change with their keyword arguments and arguements
def graphPatternReplacer(main_module:Union[GraphModule,nn.Module,callable],pattern_module:Union[GraphModule,nn.Module,callable],replace_module:Union[GraphModule,nn.Module,callable]):
    if type(main_module)!=GraphModule:
        main_module=symbolic_trace(main_module)
    if type(pattern_module)!=GraphModule:
        pattern_module=symbolic_trace(pattern_module)
    # if type(replace_module)!=GraphModule:
    #     replace_module=symbolic_trace(replace_module)
    # print(pattern_module)
    # print(replace_module)
    pattern_nodes=[]
    number_of_input=0
    number_of_output=0
    for node in pattern_module.graph.nodes:
        if node.op=='placeholder':
            number_of_input+=1
        elif node.op =='output':
            number_of_output=len(node.args)
        else:
            pattern_nodes.append(node)
    # singleNode
    if (number_of_input ==1 and number_of_output==1) or  (len(pattern_nodes)==1):
        matches = straight_chain_searcher(main_module,pattern_module)
        _replace_all_matches(main_module,matches,replace_module)
    return main_module

#put composite modules first, then primary module
_unsupported_module_dict={
    SEModule() : nn.Identity(),
    SEModule1() : nn.Identity(),
    nn.ReLU(inplace=True):nn.ReLU(),
    nn.Dropout(inplace=True):nn.Dropout(),
    nn.Hardswish():nn.ReLU(),
    nn.ReLU6():nn.ReLU(),
    nn.GELU():nn.ReLU(),
    nn.SiLU():nn.ReLU(),
    nn.LeakyReLU():nn.ReLU(),
    nn.Hardsigmoid():nn.ReLU(),
    Focus():ConvBNRModule(3,12,(5,5),(2,2),2),
    nn.Upsample():replace_resize_with_scale_factor,
    nn.MaxPool2d(3):replace_maxpool2d_k_gt5,
    nn.AvgPool2d(3):replace_avgpool2d_k_gt5,
}

def _is_replacable(pattern:Union[GraphModule,nn.Module,callable]):
    if not isinstance(pattern,GraphModule):
        pattern=symbolic_trace(pattern)
    #TODO
    return True

def replace_unsuppoted_layers(model:nn.Module,replacement_dict:Dict[Any,Union[nn.Module,callable]]=None):
    replacement_dict = replacement_dict or _unsupported_module_dict
    model=deepcopy(model)
    for pattern, replacement in replacement_dict.items():
        if isfunction(replacement):
            model=replacement(model)
        else:
            if pattern.__class__.__name__ in dir(nn):
                pattern= InstaModule(pattern)
            model=graphPatternReplacer(model,pattern,replacement)
    return model


def get_replacement_dict_default():
    return _unsupported_module_dict

