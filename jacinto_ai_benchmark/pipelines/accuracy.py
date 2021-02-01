import progiter
from .. import utils

def run(pipeline_config):
    session = pipeline_config['session']
    import_model(session, pipeline_config)
    output_list = infer_frames(session, pipeline_config)
    result = evaluate(session, pipeline_config, output_list)
    return result


def import_model(session, pipeline_config):
    calibration_dataset = pipeline_config['calibration_dataset']
    preprocess = pipeline_config['preprocess']

    calib_data = []
    num_frames = len(calibration_dataset)
    progress_bar = progiter.ProgIter(desc='data reading for calibration', total=num_frames, verbose=1)
    progress_bar.begin()
    for data_index in range(num_frames):
        data = calibration_dataset[data_index]
        data = _sequential_pipeline(preprocess, data)
        calib_data.append(data)
        progress_bar.step(inc=1)

    print('model import and calibration in progress...')
    session.import_model(calib_data)


def infer_frames(session, pipeline_config):
    input_dataset = pipeline_config['input_dataset']
    preprocess = pipeline_config['preprocess']
    postprocess = pipeline_config['postprocess']

    output_list = []
    num_frames = len(input_dataset)
    progress_bar = progiter.ProgIter(desc='model inference in progress', total=num_frames, verbose=1)
    progress_bar.begin()
    for data_index in range(num_frames):
        data = input_dataset[data_index]
        data = _sequential_pipeline(preprocess, data)
        output = session.infer_frame(data)
        output = _sequential_pipeline(postprocess, output)
        output_list.append(output)
        progress_bar.step(inc=1)

    return output_list


def evaluate(session, pipeline_config, output_list):
    # if metric is not given use input_dataset
    if 'metric' in pipeline_config and callable(pipeline_config['metric']):
        metric = pipeline_config['metric']
        metric_options = {}
    else:
        metric = pipeline_config['input_dataset']
        metric_options = pipeline_config.get('metric', {})
    #
    metric = utils.as_tuple(metric)
    metric_options = utils.as_tuple(metric_options)
    output_dict = {}
    for m, m_options in zip(metric, metric_options):
        output = m(output_list, **m_options)
        output_dict.update(output)
    #
    return output_dict


def _sequential_pipeline(pipeline, data):
    if pipeline is not None:
        pipeline = utils.as_tuple(pipeline)
        for p in pipeline:
            data = p(data)
        #
    #
    return data


def _parallel_pipeline(pipeline, data):
    if pipeline is not None:
        d_list = []
        for p in pipeline:
            data = p(data)
            d_list.append(data)
        #
        data = d_list
    #
    return data
