import os.path as osp

def relative(relative_path):    
    return osp.join(osp.dirname(osp.abspath(__file__)), relative_path)