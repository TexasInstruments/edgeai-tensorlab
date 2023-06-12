class AddModule(nn.Module):
    def forward(self, x, y):
        return x + y

class MultModule(nn.Module):
    def forward(self, x, y):
        return x * y

import operator

def replace_add_function(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m = symbolic_trace(AddModule())
    replace_m = symbolic_trace(MultModule())
    matches= xao.surgery.replacer.straight_chain_searcher(traced_m,pattern_m)
    main_modules=dict(traced_m.named_modules())

    #replace using call_function
    for start,end in matches:
        with traced_m.graph.inserting_after(start):
            new_node= traced_m.graph.call_function(operator.mul,start.args,start.kwargs) 
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)

    traced_m.recompile()
    return m


def replace_add_module(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m = symbolic_trace(AddModule())
    replace_m = symbolic_trace(MultModule())
    matches= xao.surgery.replacer.straight_chain_searcher(traced_m,pattern_m)
    main_modules=dict(traced_m.named_modules())

    # replace using call_module
    i=0
    for start,end in matches:
        traced_m.add_module(f'replace{i}',copy.deepcopy(replace_m))
        with traced_m.graph.inserting_before(start):
            new_node= traced_m.graph.call_module(f'replace{i}',start.args,start.kwargs)
            start.replace_all_uses_with(new_node)
        traced_m.graph.erase_node(start)
        i+=1
    traced_m.recompile()
    return m


def replace_conv2d(m:nn.Module):
    traced_m =  symbolic_trace(m)
    pattern_m= xao.surgery.custom_module.InstaModule(nn.Conv2d(3,16,5))
    replace_m= nn.Sequential(
                                nn.Conv2d(5,8,3),
                                nn.BatchNorm2d(8),
                                nn.Conv2d(8,20,1)
                             )
    traced_pattern= symbolic_trace(pattern_m)
    matches= xao.surgery.replacer.straight_chain_searcher(traced_m,traced_pattern)
    main_modules=dict(traced_m.named_modules())
    # pattern_modules= dict(pattern_m.named_modules())
    #filter Out the proper matched one
    #replace all filtered matches
    for start,end in matches:
        if main_modules[start.target].kernel_size!=5:
            continue
        replacement=copy.deepcopy(replace_m)
        #after adjustment
        main_in_channels = main_modules[start.target].in_channels
        main_out_channels = main_modules[end.target].out_channels
        replacement[0].in_channels=main_in_channels
        replacement[2].out_channels=main_out_channels
        replacement[2].in_channels=replacement[1].num_features=replacement[0].out_channels=math.floor(main_in_channels*(8/5))
        parent_name,name= xao.surgery.replacer._get_parent_name(start.target)
        main_modules[parent_name].__setattr__(name,replacement)
        main_modules.update({start.target,replacement})
    traced_m.recompile()
    return traced_m

