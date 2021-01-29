def run(self, **kwargs):
    import_model(self, **kwargs)
    output_list = infer_frames(self)
    metric_output = eval(self, output_list)
    return metric_output


def import_model(self):
    calibration_dataset = self.kwargs['calibration_dataset']
    preprocess = self._as_list(self.kwargs['preprocess'])

    calib_data = []
    for data in calibration_dataset:
        data = preprocess(data)
        calib_data.append(data)

    self.import_model(calib_data)


def infer_frames(self):
    input_dataset = self.kwargs['input_dataset']
    preprocess = self._as_list(self.kwargs['preprocess'])
    postprocess = self._as_list(self.kwargs['postprocess'])

    output_list = []
    for data in input_dataset:
        data = _sequential_pipeline(preprocess, data)
        output = self.infer_frame(data)
        output = _sequential_pipeline(postprocess, output)
        output_list.append(output)

    return output_list


def eval(self, output_list):
    metric = self._as_list(self.kwargs['metric'])
    output = metric(output_list)
    return output


def _sequential_pipeline(pipeline, data):
    for p in pipeline:
        data = p(data)

    return data


def _parallel_pipeline(pipeline, data):
    d_list = []
    for p in pipeline:
        data = p(data)
        d_list.append(data)

    return data
