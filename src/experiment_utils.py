import copy
import itertools


# def nstr(a):
#     return None if a.lower() == "none" else str(a)
# def nint(a):
#     return None if a.lower() == "none" else int(a)
# def uargs_to_dict(uargs):
#     return dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs])

def dict_product(data, product_keys=None):
    if product_keys is None:
        product_keys = {k for k, v in data.items() if isinstance(v, list)}
    data = {k: (v if k in product_keys else [v]) for k, v in data.items()}
    # data = {key: (val if isinstance(val, list) else [val]) for key, val in data.items()}
    return [dict(zip(data, vals)) for vals in itertools.product(*data.values())]


def align_configs(cfgs, default_cfg, prune=True):
    cfgs = copy.deepcopy(cfgs)
    for k in default_cfg.keys():  # make sure all cfgs have the default keys
        for cfg in cfgs:
            if k not in cfg:
                cfg[k] = default_cfg[k]
    # assert all(c.keys() == default_config.keys() for c in configs)
    if prune:  # prune away keys where all cfgs have the default val
        for k in default_cfg.keys():
            if all([cfg[k] == default_cfg[k] for cfg in cfgs]):
                for cfg in cfgs:
                    del cfg[k]
    return cfgs


def create_arg_list(cfg):
    def format_value(v):
        return f'"{v}"' if isinstance(v, str) else str(v)

    arg_list = []
    for key, val in cfg.items():
        if isinstance(val, list):
            arg_list.append(f'--{key} {" ".join([format_value(v) for v in val])}')
        else:
            arg_list.append(f"--{key}={format_value(val)}")
    return arg_list


def create_commands_from_arg_lists(arg_lists, prefix=None):
    n_coms, n_args = len(arg_lists), len(arg_lists[0])
    arg_lens = [max([len(arg_lists[i_com][i_arg]) for i_com in range(n_coms)]) for i_arg in range(n_args)]
    commands = [" ".join([arg_lists[i_com][i_arg].ljust(arg_lens[i_arg]) for i_arg in range(n_args)]) for i_com in
                range(n_coms)]
    if prefix is not None:
        commands = [f"{prefix} {com}" for com in commands]
    return commands


def create_commands(cfgs, default_cfg=None, prune=True, prefix=None, out_file=None):
    if default_cfg is not None:
        cfgs = align_configs(cfgs, default_cfg, prune=prune)
    assert all([set(cfg.keys()) == set(cfgs[0].keys()) for cfg in cfgs])  # make sure all cfgs have the same keys
    arg_lists = [create_arg_list(config) for config in cfgs]
    commands = create_commands_from_arg_lists(arg_lists, prefix=prefix)
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write("\n".join(commands) + "\n")
    return commands
