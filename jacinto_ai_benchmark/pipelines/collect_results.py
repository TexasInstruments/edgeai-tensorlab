import os
import glob

def collect_results(workDir):
    results = []
    logs = glob.glob(f'{workDir}/*/run.log')
    for log in logs:
        with open(log) as log_fp:
            log_data = [line for line in log_fp]
            result = log_data[-1].rstrip()
            result = result if 'BenchmarkResults' in result else ''
            results.append(result)
        #
    #
    results = sorted(results)
    with open(f'{workDir}/results.log','w') as writer_fp:
        for rline in results:
            writer_fp.write(f'{rline}\n')
        #
    #
    return results