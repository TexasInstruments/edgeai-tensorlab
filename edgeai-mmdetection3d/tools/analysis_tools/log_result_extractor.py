from matplotlib import pyplot as plt
import os
import warnings

def get_result(lines, phrase, ignore_keys=None):
    result =   {}
    ignore_keys = ignore_keys or []
    for line in lines:
        if phrase in line:
            i = line.index(phrase)
            line = line[i+len(phrase):]
        else: 
            continue
        line = line.strip()
        line = line.split(':')
        line = [x for x in line if x != '']
        line = [x.strip() for x in line]
        line = [x.split(' ', 1) for x in line]
        temp  = []
        [temp.extend(x) for x in line]
        line = temp
        line = [x.strip() for x in line]
        bools = []
        for x in line:
            try:
                x = float(x)
                bools.append(1)
            except:
                bools.append(0)
        indices = [i for i, x in enumerate(bools) if x == 0]
        for i in range(len(indices)-1):
            x= indices[i]
            y = indices[i+1]
            if y == x+1:
                line[y] = line[x] + '_' + line[y]
                line[x] =  ''
            else:
                line[y-1] = ':'.join(line[(x+1):y])
                for j in range(x+1, y-1):
                    line[j] = ''
        line = [x for x in line if x != '']
        keys = line[0::2]
        vals = line[1::2]
        assert len(keys) == len(vals)
        for key, val in zip(keys, vals):
            if key in ignore_keys:
                continue
            try:
                val = float(val)
            except:
                continue
            if key not in result:
                result[key] = []
            result[key].append(val)
    return result

def extract_result(log_path):
    result = {}
    ignore_keys = ['time', 'eta', 'memory', 'data_time']
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if ' mmengine - INFO - Epoch' in line]
        train_lines = [line for line in lines if ' mmengine - INFO - Epoch(train)' in line]
        val_lines = [line for line in lines if ' mmengine - INFO - Epoch(val)' in line]
        test_lines = [line for line in lines if ' mmengine - INFO - Epoch(test)' in line]
        result.update(get_result(train_lines,']  ', ignore_keys))
        result.update(get_result(val_lines, ']    ', ignore_keys), )
        result.update(get_result(test_lines,'] ', ignore_keys))
    except FileNotFoundError:
        warnings.warn(f'log file {log_path} not found')
    return result

def generate_plot(log_path):
    results = extract_result(log_path)
    if len(results) == 0:
        warnings.warn(f'log file {log_path} not found')
        return
    plot_results(results, log_path)
    return

def plot_results(results, log_path,  plot_key=None):
    if plot_key is None:
        plot_key = list(results.keys())
    if isinstance(plot_key, str):
        plot_key = [plot_key]
    assert isinstance(plot_key,(list, tuple)),\
        'plot_key must be a list or tuple or a single key (str)'
    path, _ = os.path.split(log_path)
    path = os.path.join(path, 'plots')
    os.makedirs(path, exist_ok=True)
    if len(plot_key) == 0:
        return
    temp = []
    for key in plot_key:
        if not isinstance(key, str):
            warnings.warn(f'key must be a string, but found {key} of type {type(key).__name__}')
            continue
        if key not in results:
            warnings.warn(f'key {key} not found in results')
            continue
        temp.append(key)
    
    plot_key = temp
    for key in plot_key:
        plt.plot(results[key])
        plt.title(key)
        plt.savefig(os.path.join(path, f'{key.replace("/", "__")}.png'))
        plt.close()

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args() if args is None else parser.parse_args(args)
    generate_plot(args.log_file)

if __name__ == '__main__':
    # main(['work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_ps-mono3d_tidl_3classes/20250710_131005/20250710_131005.log'])
    main()
    