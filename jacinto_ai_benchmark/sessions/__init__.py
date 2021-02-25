import copy
from .tvm_dlr_session import TVMDLRSession
from .tflite_rt_session import TFLiteRTSession


session_name_to_type_dict = {
    'tvmdlr' : TVMDLRSession,
    'tflitert': TFLiteRTSession,
}


session_type_to_name_dict = {
    TVMDLRSession : 'tvmdlr',
    TFLiteRTSession : 'tflitert'
}


def convert_session_names_to_types(session_type_dict):
    session_type_out = copy.deepcopy(session_type_dict)
    for k, v in session_type_dict.items():
        if isinstance(v, str):
            session_type_out[k] = session_name_to_type_dict[v]
        #
    #
    return session_type_out
