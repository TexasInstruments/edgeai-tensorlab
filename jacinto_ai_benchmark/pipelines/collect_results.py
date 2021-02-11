import os
import glob


def collect_results(work_dir):
    results = []
    logs_dirs = glob.glob(f'{work_dir}/*')
    for log_dir in logs_dirs:
        if os.path.isdir(log_dir):
            log_file = f'{log_dir}/run.log'
            with open(log_file) as log_fp:
                log_data = [line for line in log_fp]
                if len(log_data) > 0:
                    result = log_data[-1].rstrip()
                    result = result if 'Benchmark' in result else None
                    if result is not None and 'inference_path' not in results:
                        results.append(f"{result}, 'inference_path': {log_file}")
                    #
                else:
                    results.append(f"BenchmarkResults: None, 'inference_path': {log_file}")
                #
            #
        #
    #
    results = sorted(results)
    with open(f'{work_dir}/results.log','w') as writer_fp:
        for rline in results:
            writer_fp.write(f'{rline}\n')
        #
    #
    return results