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
