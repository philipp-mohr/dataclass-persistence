from __future__ import annotations

from pathlib import Path

from pytest import fixture

from dataclass_persistence import PersistentDataclass
from dataclasses import dataclass
import numpy as np


@dataclass
class NestedSomeData(PersistentDataclass):
    parameter_a: str
    array: np.ndarray

@dataclass
class SomeData(PersistentDataclass):
    parameter_a: str
    array: np.ndarray
    nested: NestedSomeData


@fixture
def file_dir():
    return Path(__file__).parent.joinpath('cache')


def test_future_import(file_dir):
    data = SomeData('my_string', np.array([0, 0]), NestedSomeData('my_string', np.array([0, 0])))
    file = file_dir.joinpath('my_file')
    data.store_to_disk(file)
    data_reconstructed = SomeData.load_from_disk(file)

    assert data_reconstructed.nested.parameter_a == data.nested.parameter_a
