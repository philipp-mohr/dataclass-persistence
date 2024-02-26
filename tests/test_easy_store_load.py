import numpy as np

from dataclass_persistence import store, load


def test_store_load():
    instance = np.array([1, 2, 3])
    store(instance, 'cache/data')
    res = load('cache/data')
    assert np.allclose(instance, res)


def test_store_load_dict():
    instance = np.array([1, 2, 3])
    store({'a': instance, 'b': instance}, 'cache/data')
    res = load('cache/data')
    assert np.allclose(instance, res['a'])


def test_store_load_dict_using_cplx_or_tuples():
    instance = {'a': 1+3j, 'b': (3, 4), 'c': ((1,2), (3,4)), 'd': [1,3,[3,4]]}
    store(instance, 'cache/data')
    res = load('cache/data')
    assert instance == res