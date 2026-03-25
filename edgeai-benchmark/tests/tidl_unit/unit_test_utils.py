from edgeai_benchmark import *
import re
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

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

def remove_dir(dir_path):
    if not os.path.isdir(dir_path):
        return -1
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        return -1

    return 0

def generate_plot(results, output_dir, plot_name, save_image = True, max_points=10000):
    """
    Generate a figure with multiple subplots, one for each output.
    
    Args:
        results: Test results dict
        output_dir: Directory to save the plots
        plot_name: Base name for the output files
        save_image: Save plot as png
        max_points: Maximum number of points to plot per output (to avoid overcrowding)
    
    Returns:
        tuple: (plot_path, plot_base64) for the combined plot
    """
    os.makedirs(output_dir, exist_ok=True)

    reference_outputs = results['expected_outputs']
    actual_outputs = results['outputs']
    nmse = results['nmse']
    mse = results['mse']
    delta = results['delta']

    # Ensure outputs are in list form
    if not isinstance(reference_outputs, list):
        reference_outputs = [reference_outputs]
    if not isinstance(actual_outputs, list):
        actual_outputs = [actual_outputs]
    
    num_outputs = len(reference_outputs)
    
    # Calculate grid dimensions for subplots
    if num_outputs <= 2:
        rows, cols = 1, num_outputs
    else:
        cols = min(3, num_outputs)  # Maximum 3 columns
        rows = (num_outputs + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False)
    
    # Process each output
    for i, (reference, actual) in enumerate(zip(reference_outputs, actual_outputs)):
        # Calculate row and column for this subplot
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        
        # Flatten arrays for scatter plot
        ref_flat = reference.flatten()
        act_flat = actual.flatten()
        
        # If there are too many points, sample a subset
        if len(ref_flat) > max_points:
            indices = np.random.choice(len(ref_flat), max_points, replace=False)
            ref_flat = ref_flat[indices]
            act_flat = act_flat[indices]
        
        # Calculate min and max for this output
        min_val = min(ref_flat.min(), act_flat.min())
        max_val = max(ref_flat.max(), act_flat.max())
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin
        
        # Plot scatter points
        scatter = ax.scatter(ref_flat, act_flat, alpha=1.0, s=10, color='red')
        
        # Plot the 45-degree line (y=x) for reference
        ax.plot([min_val, max_val], [min_val, max_val], color='blue', alpha=0.7, linewidth=2)
        
        # Set equal scales and limits
        ax.set_aspect('equal')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Add labels and title
        ax.set_xlabel('Reference Output Values')
        ax.set_ylabel('Actual Output Values')
        ax.set_title(f'Output {i}')
        ax.grid(True, alpha=0.3)
        
        # Add metrics to the plot if available
        legend_text = []
        if nmse is not None and i < len(nmse) and nmse[i] is not None:
            legend_text.append(f'NMSE: {nmse[i]:.7f}')
        if mse is not None and i < len(mse) and mse[i] is not None:
            legend_text.append(f'MSE: {mse[i]:.7f}')
        if delta is not None and i < len(delta) and delta[i] is not None:
            legend_text.append(f'Delta: {delta[i]:.7f}')
        
        if legend_text:
            # Add a text box with metrics in the top left corner
            metrics_text = '\n'.join(legend_text)
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide any unused subplots
    for i in range(num_outputs, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    # Add overall title and adjust layout
    fig.suptitle(f'Reference vs Actual Outputs - {plot_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    # Save the plot
    plot_img_path = os.path.join(output_dir, f"{plot_name}_plot.png")
    plot_base64_path = os.path.join(output_dir, f"{plot_name}_base64.txt")

    if (save_image):
        plt.savefig(plot_img_path, dpi=50)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=50)
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    with open(plot_base64_path, "w+") as f:
        f.write(f"{plot_base64}")    
    plt.close(fig)
    
    return (plot_img_path, plot_base64_path)
