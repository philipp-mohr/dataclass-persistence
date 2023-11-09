import os
from dataclasses import dataclass
from pathlib import Path

from pytest import fixture

from dataclass_persistence import Persistent


@fixture
def file_dir():
    path = Path(__file__).parent.joinpath('cache')
    if not path.exists():
        os.mkdir(path)
    return path


@dataclass
class tclass(Persistent):
    text1: str
    text2: str


# fixes issue mentioned in https://github.com/philipp-mohr/dataclass-persistence/issues/2
def test_2(file_dir):
    t = tclass(text1='AA', text2='BB', )
    file = file_dir.joinpath("testpersistence")
    print(t)
    t.store(file, mode='json')
    t_loaded = t.load(file)
    assert t_loaded == t


def test_2_with_mode_as_string(file_dir):
    t = tclass(text1='AA', text2='BB')
    file = file_dir.joinpath("testpersistence")
    print(t)
    t.store(file, mode='json')
    t_loaded = t.load(file)
    assert t_loaded == t
    t.store(file, mode='zip')
    t_loaded = t.load(file)
    assert t_loaded == t


import numpy as np


@dataclass
class MyClass(Persistent):
    my_array: np.ndarray


def test_complex_datatype_json():
    try:
        os.remove("cache/test_file.json")
    except FileNotFoundError:
        pass
    instance = MyClass(my_array=np.array([1 + 0.5j, 2 + 0.5j, 3], dtype=np.complex128))
    instance.store("cache/test_file", mode="json")
    reconstructed = MyClass.load("cache/test_file.json")
    assert np.allclose(instance.my_array, reconstructed.my_array)
    reconstructed2 = MyClass.load("cache/test_file")
    assert np.allclose(instance.my_array, reconstructed2.my_array)

    instance.store("cache/test_file", mode="zip")
    reconstructed = MyClass.load("cache/test_file")
    assert np.allclose(instance.my_array, reconstructed.my_array)
    reconstructed2 = MyClass.load("cache/test_file")
    assert np.allclose(instance.my_array, reconstructed2.my_array)