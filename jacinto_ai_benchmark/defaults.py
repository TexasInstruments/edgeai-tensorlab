import os

############################################################
# define a few commonly used defautls here

num_frames = 10 #50000
num_frames_calibration = 10 #100
bias_calibration_iterations = 10 #50
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
cuda_devices = None #[0,1,2,3] #None
tidl_dir = './dependencies/c7x-mma-tidl'
tidl_tensor_bits = 32

imagenet_train_cfg = dict(path=f'{datasets_path}/imagenet/train', split=f'{datasets_path}/imagenet/train.txt',
                          shuffle=True,num_frames=num_frames_calibration)
imagenet_val_cfg = dict(path=f'{datasets_path}/imagenet/val', split=f'{datasets_path}/imagenet/val.txt',
                        shuffle=True,num_frames=num_frames)

input_mean_imagenet = (123.675, 116.28, 103.53)
input_scale_imagenet = (0.017125, 0.017507, 0.017429)

input_mean_127p5 = (127.5, 127.5, 127.5)
input_scale_127p5 = (1/127.5, 1/127.5, 1/127.5)


