import torch
from inspect import getmodule, isfunction
from torch import nn,fx,Tensor
from torch.fx import symbolic_trace,GraphModule, Node
from typing import Dict, Any, Union, List
import operator
from copy import deepcopy
from . import custom_modules

'''
this module's function are implemented to changes nodes only.
no change is made on the incoming arguments and keyword arguments.
to do so self-made surgery functions are required.
for example, custom_surgery_functions module can be checked
'''

def _get_parent_name(target:str):
    '''gets the name of the parent module and attribute name of the module from the target of the module'''
    *parent, name = target.rsplit('.', 1)
    return ( parent[0] if parent else ''), name

def replace_module_nodes(model,pattern,replacement):
    '''replaces a  modules of pattern type to replacement module in the module structure'''
    modules = dict(model.named_modules())
    if type(replacement)==type:
        replace_obj=replacement()
    else: replace_obj=replacement
    if type(pattern)!=type:
        pattern_type=type(pattern)
    else: pattern_type=pattern
    n=0
    for key_name, module in modules.items():
        if isinstance(module,pattern_type):
            n+=1
            parent_name, name= _get_parent_name(key_name)
            replace_obj = deepcopy(replace_obj)
            modules[key_name] = replace_obj
            modules[parent_name].__setattr__(name, modules[key_name])
    print(pattern_type.__name__,n)


def replace_function_nodes(model,pattern_function,replacement,kwargs=None):
    '''replaces a call function node to node with replacement function '''
    traced_model= symbolic_trace(deepcopy(model))
    no_of_module=0
    n=0
    for node in traced_model.graph.nodes:
        if node.target==pattern_function:
            kwargs=kwargs or node.kwargs
            with traced_model .graph.inserting_before(node):
                if isfunction(replacement):
                    new_node = traced_model.graph.call_function(replacement, node.args, kwargs)
                else:
                    if type(replacement)==type:
                        replace_obj=replacement()
                    else: replace_obj=replacement
                    if type(replace_obj) != nn.Module:
                        return traced_model
                    new_node_name=type(replace_obj).__name__+str(no_of_module)
                    n+=1
                    no_of_module+=1
                    traced_model.add_submodule(new_node_name,replace_obj)
                    args=[]
                    for arg in node.args:
                        if type(arg) == Node:
                            args.append(arg)
                    new_node=traced_model.graph.call_module(new_node_name,tuple(args),{})
                node.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            traced_model.graph.erase_node(node)
    traced_model.graph.lint()
    traced_model.recompile()    
    print(pattern_function,str(n+no_of_module))
    return traced_model

#checks whether two nodes are  equal or not
def _are_both_node_equal(first_node:Node,second_node:Node,first_graoh_module:Union[GraphModule,None]=None,second_graph_module:Union[GraphModule,None]=None):
    '''
    checks whether two nodes are equal or not.
    till now for two node to be same they must have same operation
    '''
    
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
        if first_node.op == 'get_attr':          
            _,target1=_get_parent_name(first_node.target)  
            _,target2=_get_parent_name(second_node.target)  
            return target1 == target2
        
        if (first_graoh_module==None) or (second_graph_module==None):
            print("\nGraphModules are required for both nodes\nas at least one of them is 'call_module' node.")
            return None
        #for call_module node
        modules_in_1st_graph = dict(first_graoh_module.named_modules())
        modules_in_2nd_graph = dict(second_graph_module.named_modules())
        module_1=modules_in_1st_graph[first_node.target]
        module_2=modules_in_2nd_graph[second_node.target]
        #TODO change equality module fpr hyperparameters
        #both should be instances of same class
        return str(type(module_1))== str(type(module_2))
    
    #TODO add for comparing two nodes if they refer to a pair functional and module node

    return False


#searches pattern with on single input and one output other wise a single node with out checking kwargs
def straight_chain_searcher(main_module:GraphModule,pattern_module:GraphModule):
    '''
    searches for straight pattern matches in node list of the graph 
    
    it only allows:
        i)  if pattern has one input and one output
        ii) if pattern has only one node other than placeholders or output

    '''
    
    main_module_nodes:list[fx.Node] =list(main_module.graph.nodes)
    pattern_module_nodes=[]
    number_of_input=0
    number_of_output=0

    # removes placeholders or outputs
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

    # similar approach to searching pattern in an list
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
    '''
    replaces nodes from start node to end node with the replacement module

    if pattern has only a single operational node start and end wull be same
    then, it will check for replacement if it also has a single node.
    if it has only a single operational node, it will check if it is call function or call method node.
    if it is so, it will add respective node not a call module node
    '''
    
    main_modules=dict(main_module.named_modules())
    
    #if pattern has a single operational node only 
    if start == end:
            #if start is a call function or call method node
            if start.op in ['call_function','call_method']:
                traced_replacement=symbolic_trace(replace_module)
                replcament_nodes=[]

                #removes all placeholders and output node
                for node in traced_replacement.graph.nodes:
                    if node.op not in ['placeholder','output']:
                        replcament_nodes.append(node)

                #if replacement has single operational node
                if len(replcament_nodes) == 1:
                    #if operation of replacement is call function
                    if  replcament_nodes[0].op == 'call_function':
                        with main_module.graph.inserting_after(start):
                            new_node= main_module.graph.call_function(replcament_nodes[0].target,start.args,start.kwargs)
                            start.replace_all_uses_with(new_node)
                        main_module.graph.erase_node(start)
                        main_module.recompile()
                        return

                    elif replcament_nodes[0].op == 'call_method':
                    #if operation of replacement is call method
                        with main_module.graph.inserting_after(start):
                            new_node= main_module.graph.call_method(replcament_nodes[0].target,start.args,start.kwargs)
                            start.replace_all_uses_with(new_node)
                        main_module.graph.erase_node(start)
                        main_module.recompile()
                        return
                        
    #for call module start node, even pattern and replacement have a single operational node each

    if (start.op=='call_module'):
        parent_name,name =_get_parent_name(start.target)
        parent_module=main_modules[parent_name]
        parent_module.__setattr__(name,replace_module)
        main_modules[start.target]=replace_module
        newNode=start
        ptr=start.next

        #if pattern has more than one operational nodes, removes all extra nodes
        if start!=end:
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

            ptr.replace_all_uses_with(newNode)
            main_module.graph.erase_node(end)
    
    # if start of the pattern is call function or call method node,
    # even if replacement has single call_module node,
    # but pattern must have more than one operational nodes
    else:
        #creates a module for replacement and adds it to main model
        new_node_name='replaced_'+str(replace_module.__class__.__name__)+'_'+str(no_of_module_replaced)
        main_module.add_module(new_node_name,replace_module)

        # creates new node for replacement and deletes all nodes from start to end 
        with main_module.graph.inserting_before(start):
            args=[]
            for arg in start.args:
                if type(arg) == Node:
                    args.append(arg)
            newNode = main_module.graph.call_module(new_node_name,tuple(args),{})
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

            ptr.replace_all_uses_with(newNode)
            main_module.graph.erase_node(end)
        main_modules.update({new_node_name:replace_module})

    main_module.graph.lint()
    main_module.recompile()


#replaces all matches with call_module node
def _replace_all_matches(main_module:GraphModule,matches,replace_module:nn.Module):
    '''
    replace all pattern from matches with a copy of replacement module
    '''

    no_of_module_replaced=0
    for (start,end) in matches:
        replace_module_copy =deepcopy(replace_module)
        _replace_pattern(main_module,start,end,replace_module_copy,no_of_module_replaced)
        no_of_module_replaced+=1


#replace nodes if they don't need any change with their keyword arguments and arguements
def graph_pattern_replacer(main_module:Union[GraphModule,nn.Module,callable],pattern_module:Union[GraphModule,nn.Module,callable],replace_module:Union[GraphModule,nn.Module,callable]):
    '''
    searches for all matches in the graph and replaces all of them with replacement module  
    '''

    if type(main_module)!=GraphModule:
        main_module=symbolic_trace(main_module)
    if type(pattern_module)!=GraphModule:
        pattern_module=symbolic_trace(pattern_module)

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

    # pattern should have a single node or (single input and single output)
    if (number_of_input ==1 and number_of_output==1) or  (len(pattern_nodes)==1):
        matches = straight_chain_searcher(main_module,pattern_module)
        _replace_all_matches(main_module,matches,replace_module)
        print(type(pattern_module).__name__, len(matches))
    else:
        print(''' unable to change model as pattern does n't satisfy for the criteria of pattern searcher''')
    
    return main_module




