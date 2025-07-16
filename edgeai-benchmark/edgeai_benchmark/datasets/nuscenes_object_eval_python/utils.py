import io as sysio

from pyquaternion import Quaternion
import numpy as np
import os


# Compute a transform matrix by given quaternion and translation vector
    
def convert_quaternion_to_matrix(quaternion: list,
                                 translation: list = None) -> list:
    result = np.eye(4)
    result[:3, :3] = Quaternion(quaternion).rotation_matrix
    if translation is not None:
        result[:3, 3] = np.array(translation)
    return result.astype(np.float32).tolist()

