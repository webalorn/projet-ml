import h5py
f = h5py.File('1008_86.h5', 'r')

def to_dict(d):
    if isinstance(d, h5py.Dataset): return d[:]
    else: return { key : to_dict(d[key]) for key in d.keys() }

data = to_dict(f)

def to_shapes(d):
    return {k:to_shapes(v) for k,v in d.items()} if isinstance(d,dict) else d.shape
data_shape = to_shapes(data)