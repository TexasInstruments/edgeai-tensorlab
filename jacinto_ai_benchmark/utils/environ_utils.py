import os

def setup_environment(tidl_dir):
    # these have to be set from the calling environment
    # setting TIDL_BASE_PATH
    # os.environ['TIDL_BASE_PATH'] = tidl_dir
    # setting LD_LIBRARY_PATH
    # import_path = f"{tidl_dir}/ti_dl/utils/tidlModelImport/out"
    # rt_path = f"{tidl_dir}/ti_dl/rt/out/PC/x86_64/LINUX/release"
    # tfl_delegate_path = f"{tidl_path}/ti_dl/tfl_delegate/out/PC/x86_64/LINUX/release"
    # ld_library_path = os.environ.get('LD_LIBRARY_PATH','.')
    # ld_library_path = f"{ld_library_path}:{import_path}:{rt_path}:{tfl_delegate_path}"
    # os.environ['LD_LIBRARY_PATH'] = ld_library_path
    # os.environ["TIDL_RT_PERFSTATS"] = "1"
    assert 'TIDL_BASE_PATH' in os.environ, 'TIDL_BASE_PATH must be set in the calling enviroment'
    assert 'LD_LIBRARY_PATH' in os.environ, 'LD_LIBRARY_PATH must be set in the calling enviroment'
