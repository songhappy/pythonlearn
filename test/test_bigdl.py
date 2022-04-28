
from unittest import TestCase
import pytest
import numpy as np
from bigdl.orca.data import XShards


class TestSparkBackend(TestCase):

    def test_partition_ndarray(self):

        data = np.random.randn(10, 4)

        xshards = XShards.partition(data)

        data_parts = xshards.rdd.collect()

        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)

if __name__ == "__main__":
    pytest.main([__file__])
