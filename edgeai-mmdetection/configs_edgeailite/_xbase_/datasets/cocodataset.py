
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
#dataset_repeats = 1
backend_args = None

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        type='RepeatDataset',
        #times=dataset_repeats,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=None)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=None),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=None))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

evaluation = dict(interval=1, metric='bbox')
