import os
import glob

def collect_results(workDir):
    results = []
    logs_dirs = glob.glob(f'{workDir}/*')
    for log_dir in logs_dirs:
        if os.path.isdir(log_dir):
            log_file = f'{log_dir}/run.log'
            with open(log_file) as log_fp:
                log_data = [line for line in log_fp]
                result = log_data[-1].rstrip()
                result = result if 'Benchmark' in result else None
                if result is not None and 'session' not in results:
                    result = f'{result} : {log_file}'
                    results.append(result)
                #
            #
        #
    #
    results = sorted(results)
    with open(f'{workDir}/results.log','w') as writer_fp:
        for rline in results:
            writer_fp.write(f'{rline}\n')
        #
    #
    return results