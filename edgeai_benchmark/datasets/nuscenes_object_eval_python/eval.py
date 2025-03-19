import io as sysio

from pyquaternion import Quaternion
from typing import Tuple, Dict, Any, List
import numpy as np
import time
import tqdm
import json
import os

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name


class NuScenesEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 data_infos: dict,
                 data_scene_infos: dict,
                 pred_boxes: dict,
                 data_ids: List,
                 config: DetectionConfig,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.data_infos = data_infos
        self.data_scene_infos = data_scene_infos
        self.pred_boxes = pred_boxes
        self.data_ids = data_ids
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Make dirs. - Need it?
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes = self.load_prediction(pred_boxes, DetectionBox, verbose=verbose)
        self.gt_boxes = self.load_gt_boxes(DetectionBox, pred_boxes, verbose=verbose)

        #assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
        #    "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens


    def load_prediction(self, pred_boxes: dict, box_cls, verbose: bool = False) -> EvalBoxes:
        # Deserialize results and get meta data.
        all_results = EvalBoxes.deserialize(pred_boxes, box_cls)

        # Check that each sample has no more than x predicted boxes.
        for sample_token in all_results.sample_tokens:
            assert len(all_results.boxes[sample_token]) <= self.cfg.max_boxes_per_sample, \
                "Error: Only <= %d boxes per sample allowed!" % self.cfg.max_boxes_per_sample

        return all_results

    def load_gt_boxes(self, box_cls, pred_boxes: dict, verbose: bool = False) -> EvalBoxes:

        if box_cls == DetectionBox:
            attribute_map = {a['token']: a['name'] for a in self.nusc.attribute}

        sample_tokens = []
        for sample_token, _ in pred_boxes.items():
            sample_tokens.append(sample_token)

        all_annotations = EvalBoxes()

        # Load annotations and filter predictions and annotations.
        for idx, sample_token in tqdm.tqdm(enumerate(sample_tokens), leave=verbose):

            # self.nusc.get() work, but it is okay to use self.data_infos except for FCOS3D
            sample = self.nusc.get('sample', sample_token)
            #sample = self.data_infos['infos'][idx]
            sample_annotation_tokens = sample['anns']

            sample_boxes = []
            for sample_annotation_token in sample_annotation_tokens:

                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                if box_cls == DetectionBox:
                    # Get label name in detection task and filter unused labels.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])
                    if detection_name is None:
                        continue

                    # Get attribute_name.
                    attr_tokens = sample_annotation['attribute_tokens']
                    attr_count = len(attr_tokens)
                    if attr_count == 0:
                        attribute_name = ''
                    elif attr_count == 1:
                        attribute_name = attribute_map[attr_tokens[0]]
                    else:
                        raise Exception('Error: GT annotations must not have more than one attribute!')

                    sample_boxes.append(
                        box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation['translation'],
                            size=sample_annotation['size'],
                            rotation=sample_annotation['rotation'],
                            velocity=self.nusc.box_velocity(sample_annotation['token'])[:2],
                            num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                            detection_name=detection_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                            attribute_name=attribute_name
                        )
                    )
                else:
                    raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

            all_annotations.add_boxes(sample_token, sample_boxes)

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

        return all_annotations


    def main(self) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """

        # No plot
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))
        """

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        """
        if render_curves:
            self.render(metrics, metric_data_list)
        """

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        #metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))

        return metrics_summary

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list
