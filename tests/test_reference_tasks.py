import numpy as np
from src.patching_filter import ref_2hop, ref_3hop

def test_2hop_truth_table():
    vals=np.array([0,1],dtype=np.int8)
    for a in vals:
      for b in vals:
       for c in vals:
        u,y=ref_2hop(np.array([a]),np.array([b]),np.array([c]))
        assert int(u[0]) == (int(a)^int(b))
        assert int(y[0]) == ((int(a)^int(b)) & int(c))

def test_3hop_truth_table():
    vals=np.array([0,1],dtype=np.int8)
    for a in vals:
      for b in vals:
       for c in vals:
        for d in vals:
         u,v,y=ref_3hop(np.array([a]),np.array([b]),np.array([c]),np.array([d]))
         assert int(u[0]) == (int(a)^int(b))
         assert int(v[0]) == ((int(a)^int(b)) & int(c))
         assert int(y[0]) == (((int(a)^int(b)) & int(c)) ^ int(d))
