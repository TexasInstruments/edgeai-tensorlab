import copy


def dict_update(src_dict, *args, inplace=False, **kwargs):
    new_dict = src_dict if inplace else copy.deepcopy(src_dict)
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        new_dict.update(arg)
    #
    new_dict.update(kwargs)
    return new_dict


def dict_merge(target_dict, src_dict, inplace=False):
    target_dict = target_dict if inplace else copy.deepcopy(target_dict)
    assert isinstance(target_dict, dict), 'destination must be a dict'
    assert isinstance(src_dict, dict), 'source must be a dict'
    for key, value in src_dict.items():
        if hasattr(target_dict, key) and isinstance(target_dict[key], dict):
            if isinstance(value, dict):
                target_dict[key] = dict_merge(target_dict[key], **value)
            else:
                target_dict[key] = value
            #
        else:
            target_dict[key] = value
        #
    #
    return target_dict


def as_tuple(arg):
    return arg if isinstance(arg, tuple) else (arg,)


def as_list(arg):
    return arg if isinstance(arg, list) else [arg]


def as_list_or_tuple(arg):
    return arg if isinstance(arg, (list,tuple)) else (arg,)

