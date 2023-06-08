from collections import defaultdict
from random import shuffle

import toml

def select_distributed_parameters(parameters, num_machines, i = 0):
    if num_machines == 1:
        return []
    for x in range(i, len(parameters)):
        t = parameters[x]
        if t[2] > 1 and num_machines % t[2] == 0:
            distributed_parameters = select_distributed_parameters(parameters, num_machines // t[2], x + 1)
            if distributed_parameters is not None:
                return [t] + distributed_parameters
    return None


def main(grid_search_toml_file, num_machines):
    configs = toml.load(grid_search_toml_file)
    num_configs = 1
    for section in configs:
        for param in configs[section]:
            num_configs *= len(configs[section][param])

    print(f"Found {num_configs} configurations. Parallel grid search will approximately take {num_configs / num_machines * 4.5/24} days.")
    # i = input("Continue? (y/n)")
    # if i != "y":
    #     return

    machine_configs = [{} for _ in range(num_machines)]
    param_ic = [(section, param, len(configs[section][param]), 0) for section in configs for param in configs[section]]
    distributed_params = select_distributed_parameters(param_ic, num_machines)
    if distributed_params is None:
        print("Cannot evenly distribute parameters across machines.")
        return

    for machine_id in range(num_machines):
        for section, param, size, ix in distributed_params:
            if section not in machine_configs[machine_id]:
                machine_configs[machine_id][section] = defaultdict(list)
            val = configs[section][param][ix]
            if val not in machine_configs[machine_id][section][param]:
                machine_configs[machine_id][section][param].append(val)
            else:
                print(f"{val} already present in machine configuration {machine_id}.")
                return
        i = -1
        while i >= -len(distributed_params):
            section, param, size, ix = distributed_params[i]
            if ix == size - 1:
                distributed_params[i] = (section, param, size, 0)
                i -= 1
            else:
                distributed_params[i] = (section, param, size, ix+1)
                break

    for machine_id in range(num_machines):
        for t in param_ic:
            if t in distributed_params:
                continue
            section, param, _, _ = t
            if section not in machine_configs[machine_id]:
                machine_configs[machine_id][section] = defaultdict(list)
            for val in configs[section][param]:
                machine_configs[machine_id][section][param].append(val)

    machine_postfixes = [chr(i+ord('a')) for i in range(num_machines)]
    for machine_postfix, machine_config in zip(machine_postfixes, machine_configs):
        with open(grid_search_toml_file.replace(".toml", f"{machine_postfix}.toml"), "w+") as f:
            toml.dump(machine_config, f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("grid_search_toml_file")
    ap.add_argument("-n", "--num_machines", type=int, default=8)

    args = ap.parse_args()
    main(args.grid_search_toml_file, args.num_machines)