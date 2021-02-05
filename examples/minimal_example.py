from dataclass_persistence import PersistentDataclass
from dataclasses import dataclass
import numpy as np


@dataclass
class SomeData(PersistentDataclass):
    parameter_a: str
    array: np.ndarray


data = SomeData('my_string', np.array([0, 0]))
file = 'my_file'
data.store_to_disk(file)
data_reconstructed = SomeData.load_from_disk(file)

test = 1