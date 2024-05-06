TBD = None

img_norm_cfg = dict(mean=TBD, std=TBD, to_rgb=TBD)
train_pipeline = TBD
test_pipeline = TBD

dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
#dataset_repeats = 1

data = dict(
    samples_per_gpu=TBD,
    workers_per_gpu=TBD,
    train=dict(
        type='RepeatDataset',
        #times=dataset_repeats,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    # using val split for test, as the actual test split doesn't have GT
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json', #_test.json',
        img_prefix=data_root + 'leftImg8bit/val/', #'leftImg8bit/test/'
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metric='bbox')
