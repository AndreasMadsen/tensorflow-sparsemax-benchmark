
import numpy as np


class CreateDataset:
    def __init__(self, dims, obs=1000, dtype=np.float32):
        self.logits = np.random.uniform(low=-3, high=3, size=(obs, dims)) \
                               .astype(dtype)

        self.labels = np.zeros((obs, dims), dtype=dtype)
        self.labels[
            np.arange(0, obs), np.random.randint(0, dims, size=obs)
        ] = 1

        self.name = "(%d, %d)" % (obs, dims)
        self.obs = obs
        self.dims = dims

all_datasets = [
    CreateDataset(10),
    CreateDataset(100),
    CreateDataset(512),
    CreateDataset(2**14, obs=20)  # 16K
]
