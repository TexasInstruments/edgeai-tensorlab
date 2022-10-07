#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################
import os
import datetime
import sys
import copy
import argparse
import yaml
import json
import re


def run_regex_py(log_lines, regex_expr):
    value = []
    for line in log_lines:
        for r_entry in regex_expr:
            r_op = r_entry['op']
            r_pattern = r_entry['pattern']
            if line is not None and r_op == 'search':
                r_grp = r_entry.get('group', 0)
                line = re.search(r_pattern, line)
                line = line.group(r_grp) if line is not None else line
            elif line is not None and r_op == 'split':
                r_idx = r_entry['index']
                line = re.split(r_pattern, line)[r_idx]
            elif line is not None and r_op == 'findall':
                r_idx = r_entry['index']
                line = re.findall(r_pattern, line)[r_idx]
            elif line is not None and r_op == 'sub':
                r_replace = r_entry['repl']
                line = re.sub(r_pattern, r_replace, line)
            #
        #
        if line:
            value.append(line)
        #
    #
    return value


def run_regex_js(log_lines, regex_expr):
    import js2py
    search_str = '''
        function f(line, pattern) {
            return line.match(pattern)
        }
        '''
    search_js_fn = js2py.eval_js(search_str)
    value = []
    for line in log_lines:
        for r_entry in regex_expr:
            r_op = r_entry['op']
            r_pattern = r_entry['pattern']
            if line is not None and line != 'null' and r_op == 'search':
                r_grp = r_entry.get('group', None)
                case_sensitive = r_entry.get('case_sensitive', True)
                line = line if case_sensitive else line.lower()
                line = search_js_fn(line, r_pattern)
                if line is not None and line != 'null' and r_grp is not None:
                    line = line[r_grp]
                    if r_entry.get('dtype',None) == 'float':
                        line = float(line)
                    elif r_entry.get('dtype',None) == 'int':
                        line = int(line)
                    #
                #
            #
        #
        if line is not None and line != 'null':
            value.append(line)
        #
    #
    return value


def write_summary(log_file_path, summary_file_path, log_summary_regex, format='py'):
    log_summary = []
    with open(log_file_path) as fp:
        log_lines = fp.readlines()
        log_lines = [l.rstrip() for l in log_lines]
    #
    # log_summary_regex can be a list
    # log_summary_regex can also be a dict with the dict keys being formats supported 'py' or 'js' or ['py', 'js']
    if isinstance(log_summary_regex, dict):
        format_found = False
        log_summary_regex_format = None
        for key in log_summary_regex.keys():
            key_split = key.split(',')
            if isinstance(key_split, (list,tuple)) and format in key_split:
                format_found = True
                log_summary_regex_format = key
            elif key == format:
                format_found = True
                log_summary_regex_format = key
            #
        #
        assert format_found, f'format {format} not found in log_summary_regex'
        log_summary_regex = log_summary_regex[log_summary_regex_format]
    #
    for regex_entry in log_summary_regex:
        regex_expr = regex_entry['regex']
        if format == 'py':
            value = run_regex_py(log_lines, regex_expr)
        elif format == 'js':
            value = run_regex_js(log_lines, regex_expr)
        else:
            assert False, f'unsupported format {format}'
        #
        if len(value) > 0:
            regex_entry['value'] = value
        #
        log_summary.append(regex_entry)
    #
    for summary_ext in ['.yaml', '.json']:
        summary_file_path_base = os.path.splitext(summary_file_path)[0]
        summary_file_path_ext = summary_file_path_base+summary_ext
        with open(summary_file_path_ext, 'w') as wfp:
            if summary_ext == '.yaml':
                yaml.dump(log_summary, wfp)
            elif summary_ext == '.json':
                json.dump(log_summary, wfp)
            #
        #
    #
    return log_summary


def main(args, config):
    training_log_file_path = config['training']['log_file_path']
    training_summary_file_path = config['training']['summary_file_path']
    training_log_summary_regex = config['training']['log_summary_regex']
    if training_log_summary_regex is not None:
        write_summary(training_log_file_path, training_summary_file_path, training_log_summary_regex, format=args.format)
    #

    compilation_log_file_path = config['compilation']['log_file_path']
    compilation_summary_file_path = config['compilation']['summary_file_path']
    compilation_log_summary_regex = config['compilation']['log_summary_regex']
    if compilation_log_summary_regex is not None:
        write_summary(compilation_log_file_path, compilation_summary_file_path, compilation_log_summary_regex, format=args.format)
    #


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--run_file_path', type=str)
    parser.add_argument('--format', type=str, default='js', choices=['js','py'])
    args = parser.parse_args()

    # read the config
    with open(args.run_file_path) as fp:
        if args.run_file_path.endswith('.yaml'):
            config = yaml.safe_load(fp)
        elif args.run_file_path.endswith('.json'):
            config = json.load(fp)
        else:
            assert False, f'unrecognized config file extension for {args.config_file}'
        #
    #

    main(args, config)