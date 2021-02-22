import os
import glob
import pickle


def collect_results(settings, work_dir):
    results = []
    logs_dirs = glob.glob(f'{work_dir}/*')
    for log_dir in logs_dirs:
        if os.path.isdir(log_dir):
            pkl_filename = f'{log_dir}/result.pkl'
            try:
                with open(pkl_filename, 'rb') as pkl_fp:
                    result = pickle.load(pkl_fp)
                    results.append(result)
                #
            except:
                pass
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