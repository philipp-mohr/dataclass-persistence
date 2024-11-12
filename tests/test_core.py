from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction

import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional

from strenum import StrEnum

from dataclass_persistence import Persistent, EXCLUDE, EXPLICIT, SEPARATE
from pytest import fixture, mark


# some example scenario, where simulation data like configuration and results are stored in nested hierarchy
def verify_load_store(c: Persistent, cls_=None):
    if cls_ is None:
        cls_ = type(c)
    json_ = c.to_json()
    c_loaded = cls_.from_json(json_)
    assert c == c_loaded  # performs nested comparison


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


@mark.parametrize("mode", ['pkl', 'json', 'zip', 'zip_json', 'zip_pkl'])
def test_store_load_sim_data(file_dir, mode):
    sim_data = SimDataCustom(config_system=ConfigSystemCustom(conf_component=ConfigSomeComponentA()),
                             sim_points=[SimPointCustom(params={'a': 1.0},
                                                        result=ResultSystemCustom(np.zeros(1000)))
                                         for i in range(100)])
    file = file_dir.joinpath('some_name')
    sim_data.store(file, mode=mode)
    sim_data_reconstructed: SimDataCustom = SimDataCustom.load(file, mode=mode)

    sim_points_rec = sim_data_reconstructed.sim_points
    sim_points = sim_data.sim_points
    for i, sim_point in enumerate(sim_points):
        assert np.all(sim_point.result.ary == sim_points_rec[i].result.ary)
    assert np.all(sim_data.config_system.conf_component.array_a ==
                  sim_data_reconstructed.config_system.conf_component.array_a)


@dataclass
class Class1(Persistent):
    param_a: str
    param_b: str = field(metadata=EXCLUDE)
    param_c: list[np.ndarray] = None


def test_do_not_store_load_excluded_fields(file_dir):
    cfg = Class1('a', 'b')
    cfg_json = cfg.to_json()
    dict_data = json.loads(cfg_json)
    assert dict_data['param_b'] is None
    cfg_loaded = Class1.load_json(cfg_json)
    assert cfg_loaded.param_b is None


@dataclass
class NestedClass(Persistent):
    cls_a: Class1 = field(metadata=EXCLUDE)
    cls_b: Class1
    cls_c: list[Class1]
    param_a: str = field(metadata=EXCLUDE)


def test_replace_fields_of_instance_which_are_not_excluded(file_dir):
    config = NestedClass(Class1('a', 'b'), Class1('c', 'd'), [Class1(param_a='a', param_b='b',
                                                                     param_c=[np.array([0])])], 'e')
    file = file_dir.joinpath('config_without_excluded_field')
    config.store(file=file)
    loaded = NestedClass.load(file=file)
    assert loaded.cls_a is None
    assert loaded.cls_b.param_b is None
    assert loaded.param_a is None
    config.update(file=file)
    assert config.cls_a == Class1('a', 'b')
    assert config.cls_b.param_b == 'd'
    assert config.param_a == 'e'
    config.update(file)
    # do not update excluded fields
    assert config.cls_c[0].param_b == 'b'
    # in cases of different list sizes, .update() shall use the loaded list.
    loaded.cls_c.append(Class1(param_a='a', param_b='b'))
    loaded.store(file)
    config.update(file)
    assert len(config.cls_c) == 2


@dataclass
class Class2(Persistent):
    a: str = 'abc'


@mark.parametrize("mode", ['pkl', 'json', 'zip', 'zip_json', 'zip_pkl'])
def test_deal_with_multiple_dots_in_file_name(file_dir, mode):
    my = Class2()
    my.store((_f := file_dir.joinpath('file_a=0.1_b=0.3')), mode=mode)
    my_loaded = Class2.load(_f, mode=mode)
    assert my == my_loaded


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
class MyImpl2(MyBase):
    a: int = 5


@dataclass
class DataclassWithSubClass(Persistent):
    val: MyBase = None
    val2: MyBase = None
    val3: list[MyBase] = None
    val4: tuple[MyBase] = None
    val5: Union[MyBase, MyImpl2] = None


def test_restore_subclass_of_abstract_base_class():
    c = DataclassWithSubClass(val=MyImpl(), val3=[MyImpl(5)], val4=(MyImpl(5),), val5=MyImpl(a=7))
    verify_load_store(c, DataclassWithSubClass)


@dataclass
class MyClassInitFalse(Persistent):
    b: int = field(init=False)
    a: int = 5


def test_restore_init_false_fields():
    my = MyClassInitFalse()
    my.b = 10
    verify_load_store(my, MyClassInitFalse)


@dataclass
class MyClass2FilterFields(Persistent):
    a: np.ndarray = field(default=None, metadata=EXPLICIT('large_data'))


@dataclass
class MyClassFilterFields(Persistent):
    a: np.ndarray = field(default=None, metadata=EXPLICIT('large_data'))
    b: MyClass2FilterFields = None
    c: list[MyClass2FilterFields] = field(default=None, metadata=EXPLICIT('large_data'))


def test_preserve_fields_only_if_explicitly_required(file_dir, request):
    # Specific fields may contain data which is irrelevant for preservation in most cases.
    # Those fields may be marked with the 'explicit' key inside inside of the metadata with some identifier
    # e.g. 'large_data'.
    # ONLY if this identifier is provided in store(explicit=['large_data',...]]), the field will be preserved.
    my = MyClassFilterFields(a=np.array([100000]),
                             b=MyClass2FilterFields(np.array([100000])),
                             c=[MyClass2FilterFields(np.array([100000]))])
    json3 = my.to_json()  # by default explicit marked fields are not stored
    my.store((_f := file_dir.joinpath(request.node.name)), explicit=['large_data'])
    json2 = my.to_json(explicit=['large_data'])
    my_loaded = MyClassFilterFields.load(_f)
    my_loaded2 = MyClassFilterFields.from_json(json2)
    my_loaded3 = MyClassFilterFields.from_json(json3)

    assert my_loaded.a is not None and my_loaded.a == my_loaded2.a
    assert my_loaded3.a is None

    assert my_loaded.b.a is not None and my_loaded.b.a == my_loaded2.b.a
    assert my_loaded3.b.a is None

    assert my_loaded.c is not None and my_loaded.c == my_loaded2.c
    assert my_loaded3.c is None


@dataclass
class NestedContainer(Persistent):
    a: np.ndarray = None


@dataclass
class MyClassLargeFields(Persistent):
    a: np.ndarray = None
    b: np.ndarray = None
    nest: NestedContainer = None


def test_save_large_arrays_in_separate_file(request, file_dir):
    # goal: only put ID inside of json file which point to some SEPARATE file which contains the data
    # numpy array with size larger than 100 are automatically stored in separate .pkl
    # data = {'a': np.arange(10000), 'b': np.arange(10000)}
    # np.savez('cache/np_file.npz', **data)
    # loaded_npz = np.load('cache/np_file.npz')
    # data_loaded = {k: loaded_npz[k] for k in loaded_npz.files}
    my = MyClassLargeFields(a=np.arange(1000000), b=np.arange(50),
                            nest=NestedContainer(np.arange(100000)))
    my.store((_f := file_dir.joinpath(request.node.name)))
    my_loaded = MyClassLargeFields.load(_f)
    assert np.all(my_loaded.a == my.a)
    assert np.all(my_loaded.b == my.b)


class MyStrEnum(StrEnum):
    opt1 = auto()
    opt2 = auto()


@dataclass
class MyDataClassWithStrEnum(Persistent):
    opt: str | MyStrEnum = MyStrEnum.opt1


def test_field_with_str_enum():
    verify_load_store(MyDataClassWithStrEnum(MyStrEnum.opt2))
    verify_load_store(MyDataClassWithStrEnum('opt2'))


class MyEnum2(Enum):
    SOME_ENUM_NAME: Fraction = Fraction(1, 3)


@dataclass
class CfgWithFraction(Persistent):
    code_rate: MyEnum2 = MyEnum2.SOME_ENUM_NAME
    number: Fraction = Fraction(1, 3)


def test_various_field_types():
    verify_load_store(CfgWithFraction())


@dataclass
class ClassWithMultiDimensional(Persistent):
    ary: np.ndarray = None
    lst: list[int] = None


def test_store_multi_dimensional_arrays(request, file_dir):
    len_array_separate = 1000
    for size in [999, 1001]:
        a = ClassWithMultiDimensional(ary=np.arange(1, size)[np.newaxis, :],
                                      lst=list(range(size)))
        a.store((f := file_dir.joinpath(request.node.name)), len_array_separate=len_array_separate)
        b = ClassWithMultiDimensional.load(f)
        assert np.all(a.ary == b.ary)
