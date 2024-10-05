TBD = None


# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
#dataset_repeats = 1

data = dict(
    samples_per_gpu=TBD,
    workers_per_gpu=TBD,
    train=dict(
        type='RepeatDataset',
        #times=dataset_repeats,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=TBD)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=TBD),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=TBD))

evaluation = dict(interval=1, metric='mAP')
