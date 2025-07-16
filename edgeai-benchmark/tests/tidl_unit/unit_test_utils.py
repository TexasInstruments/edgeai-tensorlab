from edgeai_benchmark import *
import re

def get_tidl_performance(interpreter, session_name="onnxrt"):
    '''
    Returns performance data for interpreter

    :param interpreter: Runtime session
    :param session_name: 'onnxrt','tflrt','tvmrt'
    :return: Stats dictionary containing following
                num_subgraphs: Number of TIDL Subgraphs
                total_time: Total time taken for run (ms)
                core_time: Total time taken barring the io copy time (ms)
                subgraph_time: Total TIDL Subgraphs processing time (ms)
                read_total: Total DDR Read bandwidth (MB/s) [0 for x86 runs]
                write_total: Total DDR Write bandwidth (MB/s) [0 for x86 runs]
             
    '''

    '''
    get_TI_benchmark_data returns the following in a dictionary:

    'ts:run_start'                : Start timestamp of OSRT session run start (in nanoseconds)
    'ts:run_end'                  : End timestamp of OSRT session run end (in nanoseconds)
    'ddr:read_start'              : DDR Read Bandwidth at OSRT session run start (in bytes)
    'ddr:read_end'                : DDR Read Bandwidth at OSRT session run end (in bytes)
    'ddr:write_start'             : DDR Write Bandwidth at OSRT session run start (in bytes)
    'ddr:write_end'               : DDR Write Bandwidth at OSRT session run end (in bytes)

    ts:subgraph_*_copy_in_start   : Start timestamp of TIDL Input Buffers copy (in nanoseconds)
    ts:subgraph_*_copy_in_end     : End timestamp of TIDL Input Buffers copy (in nanoseconds)
    ts:subgraph_*_copy_proc_start : Start timestamp of TIDL Graph process (in nanoseconds)
    ts:subgraph_*_copy_proc_end   : End timestamp of TIDL Graph processing (in nanoseconds)
    ts:subgraph_*_copy_out_start  : Start timestamp of TIDL Output Buffers copy (in nanoseconds)
    ts:subgraph_*_copy_out_end    : End timestamp of TIDL Output Buffers copy (in nanoseconds)

    Note1: * in subgraph stats is the subgraph name, Ex: ts:subgraph_subgraph_0_copy_in_start
    Note2: DDR stats will be 0 while running on PC.
    '''
    benchmark_dict = interpreter.get_TI_benchmark_data()
    subgraph_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            if session_name == 'onnxrt':
                value = stat.split("ts:subgraph_")
                value = value[1].split("_proc_start")
                subgraphIds.append(value[0])
            else:
                subgraphIds.append(int(re.sub("[^0-9]", "", stat)))

    for i in range(len(subgraphIds)):
        subgraph_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
    #
    copy_time = cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    total_time = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
    write_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    read_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

    # change units
    total_time = total_time / 1000000       # Conveting to miliseconds
    copy_time = copy_time / 1000000         # Conveting to miliseconds
    subgraph_time = subgraph_time / 1000000 # Conveting to miliseconds
    write_total = write_total / 1000000     # Conveting to MB/s
    read_total = read_total / 1000000       # Conveting to MB/s

    # core time excluding the copy overhead
    core_time = total_time - copy_time
    stats = {
        'num_subgraphs': len(subgraphIds),
        'total_time': total_time,
        'core_time': core_time,
        'subgraph_time': subgraph_time,
        'write_total': write_total,
        'read_total': read_total
    }
    #
    return stats