import copy


def dict_update(target_dict, **src_dict):
    new_dict = copy.deepcopy(target_dict)
    new_dict.update(src_dict)
    return new_dict


def dict_merge(target_dict, **src_dict):
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

