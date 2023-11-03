import os
from dataclasses import dataclass
from pathlib import Path

from pytest import fixture

from dataclass_persistence import Persistent, Mode


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
    t.store(file, mode=Mode.JSON)
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
