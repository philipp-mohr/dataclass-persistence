from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional

from dataclass_persistence import Persistent, EXCLUDE, Mode, EXPLICIT
from pytest import fixture, mark


# some example scenario, where simulation data like configuration and results are stored in nested hierarchy


@dataclass
class ResultSystemCustom:
    ary: np.ndarray


@dataclass
class SimPointCustom:
    params: Dict[str, object]
    result: ResultSystemCustom


@dataclass
class ConfigSomeComponentA:
    param_a: int = 1
    array_a: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3]))


@dataclass
class ConfigSomeComponentB:
    param_b: int = 2
    array_b: np.ndarray = field(default_factory=lambda: np.array([4, 5, 6]))


@dataclass
class ConfigSystemCustom:
    conf_component: Union[ConfigSomeComponentA, ConfigSomeComponentB]


@dataclass
class SimDataCustom(Persistent):
    config_system: ConfigSystemCustom
    sim_points: List[SimPointCustom]


@fixture
def file_dir():
    return Path(__file__).parent.joinpath('cache')


@mark.parametrize("mode", [Mode.ZIP, Mode.JSON])
def test_store_load_sim_data(file_dir, mode):
    sim_data = SimDataCustom(config_system=ConfigSystemCustom(conf_component=ConfigSomeComponentA()),
                             sim_points=[SimPointCustom(params={'a': 1.0},
                                                        result=ResultSystemCustom(np.zeros(1000)))
                                         for i in range(100)])
    file = file_dir.joinpath('some_name')
    sim_data.store(file, mode=mode)
    sim_data_reconstructed: SimDataCustom = SimDataCustom.load(file)

    sim_points_rec = sim_data_reconstructed.sim_points
    sim_points = sim_data.sim_points
    for i, sim_point in enumerate(sim_points):
        assert np.all(sim_point.result.ary == sim_points_rec[i].result.ary)
    assert np.all(sim_data.config_system.conf_component.array_a ==
                  sim_data_reconstructed.config_system.conf_component.array_a)


@dataclass
class Class(Persistent):
    param_a: str
    param_b: str = field(metadata=EXCLUDE)


def test_do_not_store_load_excluded_fields(file_dir):
    config = Class('a', 'b')
    file = file_dir.joinpath('config_without_private_field')
    config._store_to_disk_uncompressed_single_json_file(file=file)
    with open(str(file.with_suffix('.json')), 'r') as data_file:
        json_string = '\n'.join(data_file.readlines())
        dict_data = json.loads(json_string)
    assert dict_data['param_b'] is None
    config_loaded = Class.load(file)
    assert config_loaded.param_b is None


@dataclass
class NestedClass(Persistent):
    cls_a: Class = field(metadata=EXCLUDE)
    cls_b: Class
    cls_c: list[Class]
    param_a: str = field(metadata=EXCLUDE)


def test_replace_fields_of_instance_which_are_not_excluded(file_dir):
    config = NestedClass(Class('a', 'b'), Class('c', 'd'), [Class(param_a='a', param_b='b')],'e')
    file = file_dir.joinpath('config_without_excluded_field')
    config.store(file=file)
    loaded = NestedClass.load(file=file)
    assert loaded.cls_a is None
    assert loaded.cls_b.param_b is None
    assert loaded.param_a is None
    config.update(file=file)
    assert config.cls_a == Class('a', 'b')
    assert config.cls_b.param_b == 'd'
    assert config.param_a == 'e'
    config.update(file)
    # do not update excluded fields
    assert config.cls_c[0].param_b == 'b'


@dataclass
class ClassWithNumpy(Persistent):
    param_a: np.ndarray
    param_b: str = 'hello'


def test_dataclass_with_numpy_arrays(file_dir):
    ary = np.array([1, 2, 3])
    config = ClassWithNumpy(ary, 'b')
    file = file_dir.joinpath('config_with_numpy')
    config.store(file=file)
    config_loaded = ClassWithNumpy.load(file)

    assert np.all(config_loaded.param_a == ary)


@dataclass
class ClassWithTuple(Persistent):
    param_a: Tuple[int, int]


def test_dataclass_with_tuple(file_dir):
    tpl = (1, 2)
    config = ClassWithTuple(tpl)
    file = file_dir.joinpath('class_with_tuple')
    config.store(file=file)
    config_loaded = ClassWithTuple.load(file)

    assert np.all(config_loaded.param_a == tpl)


@dataclass
class ClassWithComplexValues(Persistent):
    param_a: Tuple[complex, int]
    list_cplx: List[complex]
    ary: np.ndarray


def test_dataclass_with_complex_value(file_dir):
    tpl = (1 + 2j, 2)
    list_cplx = [1 + 10j] * 10
    config = ClassWithComplexValues(tpl, ary=np.array([1 + 3j]), list_cplx=list_cplx)
    file = file_dir.joinpath('class_with_cplx')
    config.store(file=file)
    config_loaded = ClassWithComplexValues.load(file)
    assert np.all(config_loaded.param_a == tpl)


@dataclass(eq=True)
class ClassWithUnions(Persistent):
    param_a: Union[complex, int] = None
    param_d: Union[float, ConfigSomeComponentA] = ConfigSomeComponentA()
    param_b: Union[float, str] = 1.0
    param_c: Union[ConfigSomeComponentA, ConfigSomeComponentB] = ConfigSomeComponentA()


def test_dataclass_with_union(file_dir):
    ref = ClassWithUnions()
    ref.store(file_dir.joinpath('file_with_unions'))
    res = ClassWithUnions.load(file_dir.joinpath('file_with_unions'))
    assert ref.param_a == res.param_a
    assert ref.param_b == res.param_b
    assert ref.param_c.param_a == res.param_c.param_a
    assert ref.param_d.param_a == res.param_d.param_a


def test_dataclass_same_name_different_directory():
    from tests.data.a import SomeCls as SomeClsA
    from tests.data.b import SomeCls as SomeClsB

    @dataclass
    class SomeCls(Persistent):
        data: Union[SomeClsA, SomeClsB] = None

    file = 'cache/class_same_name_differnt_dir'
    some_cls = SomeCls(SomeClsB('3'))
    some_cls.store(file)
    some_cls_loaded = SomeCls.load(file)
    assert hasattr(some_cls_loaded.data, 'valb')


class MyEnum(Enum):
    VAL_A = 'val a'
    VAL_B = 'val b'


@dataclass
class EnumDataclass(Persistent):
    val: MyEnum = MyEnum.VAL_A


def test_enum():
    c = EnumDataclass()
    json_ = c.to_json()
    c_loaded = EnumDataclass.from_json(json_)
    assert c == c_loaded


@dataclass
class MyBase(ABC):
    pass


@dataclass
class MyImpl(MyBase):
    a: int = 5


@dataclass
class DataclassWithSubClass(Persistent):
    val: MyBase = None
    val2: MyBase = None
    val3: list[MyBase] = None
    val4: tuple[MyBase] = None


def test_restore_subclass_of_abstract_base_class():
    c = DataclassWithSubClass(val=MyImpl(), val3=[MyImpl(5)], val4=(MyImpl(5),))
    json_ = c.to_json()
    c_loaded = DataclassWithSubClass.from_json(json_)
    assert c == c_loaded  # performs nested comparison


@dataclass
class MyClassInitFalse(Persistent):
    b: int = field(init=False)
    a: int = 5


def test_restore_init_false_fields():
    my = MyClassInitFalse()
    my.b = 10
    my_json = my.to_json()
    my2 = MyClassInitFalse.from_json(my_json)
    assert my == my2


@dataclass
class MyClassFilterFields(Persistent):
    a: np.ndarray = field(default=None, metadata=EXPLICIT('large_data'))


def test_preserve_fields_only_if_explicitly_required(file_dir, request):
    # Specific fields may contain data which is irrelevant for preservation in most cases.
    # Those fields may be marked with the 'explicit' key inside inside of the metadata with some identifier
    # e.g. 'large_data'.
    # ONLY if this identifier is provided in store(explicit=['large_data',...]]), the field will be preserved.
    my = MyClassFilterFields(a=np.array([100000]))
    my.store((_f := file_dir.joinpath(request.node.name)), explicit=['large_data'])
    json2 = my.to_json(explicit=['large_data'])
    json3 = my.to_json()  # by default explicit marked fields are not stored
    my_loaded = MyClassFilterFields.load(_f)
    my_loaded2 = MyClassFilterFields.from_json(json2)
    my_loaded3 = MyClassFilterFields.from_json(json3)
    assert my_loaded.a == my_loaded2.a
    assert my_loaded3.a is None


SEPARATE = {'separate': True}


@dataclass
class MyClassLargeFields(Persistent):
    a: Optional[np.ndarray] = field(default=None, metadata=SEPARATE)


@mark.skip()
def test_separate_large_fields():
    my = MyClassLargeFields(a=np.array(100000))
    my.store()  # goal: only put ID inside of json file which point to some SEPARATE file which contains the data
