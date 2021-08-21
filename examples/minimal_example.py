from dataclass_persistence import Persistent
from dataclasses import dataclass
import numpy as np


@dataclass
class SomeData(Persistent):
    parameter_a: str
    array: np.ndarray


data = SomeData('my_string', np.array([0, 0]))
file = 'my_file'
data.store(file)
data_reconstructed = SomeData.load(file)
