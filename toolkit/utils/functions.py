import argparse


# config -> args [只把config里存在，但是args中不存在或者为None的部分赋值]
def merge_args_config(args, config):
    args_dic = vars(args)  # convert to map version
    for key in config:
        if key not in args_dic or args_dic[key] is None:
            args_dic[key] = config[key]
    args_new = argparse.Namespace(**args_dic)  # change to namespace
    return args_new


def func_update_storage(inputs, prefix, outputs):
    for key in inputs:
        val = inputs[key]
        # update key and value
        newkey = f"{prefix}_{key}"
        newval = val
        # store into outputs
        assert newkey not in outputs
        outputs[newkey] = newval
