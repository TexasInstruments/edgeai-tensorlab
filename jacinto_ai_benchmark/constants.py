
# data layout constants
NCHW = 'NCHW'
NHWC = 'NHWC'

# pipeline type constants
PIPELINE_UNDEFINED = None
PIPELINE_ACCURACY = 'accuracy'
PIPELINE_SOMETHING = 'something'

# frequency of the core C7x/MMA processor that accelerates Deep Learning Tasks
# this constant is used to convert cycles to time : time = cycles / DSP_FREQ
DSP_FREQ = 1e9

# other common constants
MILLI_CONST = 1e3 # multiplication by 1000 is to convert seconds to milliseconds
MEGA_CONST = 1e6  # convert raw data to mega : example bytes to mega bytes (MB)
GIGA_CONST = 1e9
ULTRA_CONST = 1e6

