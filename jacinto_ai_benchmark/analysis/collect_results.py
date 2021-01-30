import os
import glob

def collect_results(workDir):
    results = []
    logs = glob.glob(f'{workDir}/*/run.log')
    for log in logs:
        with open(log) as log_fp:
            log_data = [line for line in log_fp]
            perf = log_data[-2].rstrip()
            perf = perf if 'Performance' in perf else None
            acc = log_data[-1].rstrip()
            acc = acc if 'Accuracy' in acc else None
            log_dir = '/'.join(os.path.split(log)[:-1])
            results.append(f'{log_dir}, {acc}, {perf}')
        #
    #
    with open(f'{workDir}/results.log','w') as writer_fp:
        for rline in results:
            writer_fp.write(f'{rline}\n')
        #
    #
    return results