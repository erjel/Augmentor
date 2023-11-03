import unittest
import numpy as np
from augmentor.warp import Warp
from dataprovider3 import Dataset

class TestWarp(unittest.TestCase):
    
    def test_warp(self):
        aug = Warp()

        dset = Dataset()
        dset.add_data('img', np.random.rand(24,256,256).astype(np.float32))
        dset.add_data('seg', np.zeros((24,256,256), dtype=int))
        dset.add_mask('msk', np.ones((24,256,256)), loc=False)

        d = 128
        spec = dict(img=(22,d,d), seg=(24,d,d), msk=(24,d,d))

        spec2 = aug.prepare(spec, imgs=['img'])
        sample = dset(spec=spec2)
        sample = aug(sample)