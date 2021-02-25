import os
import glob
import pickle
from .. import utils


def collect_results(settings, work_dir, print_results=True):
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
    results = sorted(results, key=lambda item: item['inference_path'])
    if settings.enable_logging:
        result_filename = os.path.join(work_dir, 'results.log')
        with open(result_filename,'w') as writer_fp:
            for result in results:
                writer_fp.write(f'\n{utils.round_dict(result)}')
            #
        #
        pkl_filename = os.path.join(work_dir, 'results.pkl')
        with open(pkl_filename, 'wb') as fp:
            pickle.dump(results, fp)
        #
    #
    if print_results:
        for result in results:
            print(utils.round_dict(result))
        #
    #
    return results