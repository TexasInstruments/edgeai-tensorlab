TBD = None


# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
#dataset_repeats = 1

data = dict(
    samples_per_gpu=TBD,
    workers_per_gpu=TBD,
    train=dict(
        type='RepeatDataset',
        #times=dataset_repeats,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.txt',
            img_prefix=data_root + 'WIDER_train/',
            pipeline=None)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=None),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=None))

evaluation = dict(
    save_best='mAP', interval=1, metric='mAP')
