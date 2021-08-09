import importlib
import json
import logging
import zipfile
from dataclasses import dataclass, is_dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Union, Type, TypeVar, get_type_hints

import numpy as np

# excludes the field from being stored to file
# e.g. var_a: str = field(metadata=EXCLUDE)
# add other meta data through dictionary union operator:
# e.g. var_a: str = field(metadata=EXCLUDE|{'other_metakey': some_val})
EXCLUDE_KEY = 'exclude'
EXCLUDE = {EXCLUDE_KEY: True}


def write_string_to_file(data_str: str, file, b_logging: bool = True):
    """

    :param file:
    :param data_str:
    :param relative_folder_path: e.g. relative_folder_path='./results/
    :return:
    """
    path = Path(file).parent
    if not path.exists():
        Path.mkdir(path, parents=True)
        print('Created directory ' + str(path))
    outfile = open(file, 'w')
    if b_logging:
        logging.info('write: ' + str(file))
    outfile.write(data_str)
    outfile.close()


def if_key_exists_increment_id(dic: dict, name: str, id: int = 1):
    keys = [key for key, value in dic.items() if name in key.lower()]
    if len(keys) == 0:
        return '{}_{}'.format(name, id)
    else:
        return '{}_{}'.format(name, len(keys) + 1)


built_in_types = [int, float, str, bool, complex, type(None)]
built_in_types_names = [t.__name__ for t in built_in_types]


def dataclass_to_dicts(instance,
                       dict_heavy: Dict[str, np.ndarray] = None,
                       name_instance: str = '') -> Tuple[Union[None, dict, list, tuple, str,
                                                               float, int, bool],
                                                         Dict[str, object]]:
    """


    :param name_instance:
    :param instance: e.g. some class structure resulting in a dictionary like in the example dict_some_dataclass
    :param dict_some_dataclass: {}
    :param dict_heavy: {}
    :return:
    e.g.
    dict_some_dataclass = {
        'config_system':{
            'conf_component':{
                'class_type': 'ConfigSomeComponentA', # this info is needed for reconstruction when e.g. following type
                                                      # is used: Union[ConfigSomeComponentA, ConfigSomeComponentB]
                'param_a': int,
                'array_a': 'array_a_1'.
            }
        },
        'sim_points': [
            {'param': 1.0,
             'result': {
                    'value': 0.1
                }
            },
            {'param': 2.0,
             'result': {
                    'value': 0.2
                }
            }
        ]
    }

    dict_numpy_array = {
        'array_a_1': np.array([1,2,3])
    }
    """
    if dict_heavy is None:
        dict_heavy = {}
    json_object = {}
    if is_dataclass(instance):
        _fields = [(getattr(instance, _field.name), _field) for _field in fields(instance)]
        for value, _field in _fields:
            if _field.metadata.get(EXCLUDE_KEY, False):
                json_object[_field.name] = None
            else:
                json_object[_field.name] = dataclass_to_dicts(value,
                                                              dict_heavy,
                                                              _field.name)[0]
                conditions = [hasattr(_field.type, '__origin__') and _field.type.__origin__ == Union,
                              # required for future imports: and type(value).__name__ != _field.type
                              type(value) != _field.type and value is not None and type(value).__name__ != _field.type]
                if any(conditions):
                    if is_dataclass(value):
                        json_object[_field.name]['type'] = f'{type(value).__module__}|{type(value).__name__}'
                    elif type(value) in built_in_types:
                        json_object[_field.name] = {'type': str(type(value).__name__),
                                                    'value': json_object[_field.name]}
        return json_object, dict_heavy
    elif type(instance) == tuple:
        return tuple(
            [dataclass_to_dicts(item, dict_heavy, )[0] for item in instance]), dict_heavy
    elif type(instance) == list:
        return [dataclass_to_dicts(item, dict_heavy, )[0] for item in instance], dict_heavy

    elif type(instance) == dict:
        return {k: dataclass_to_dicts(v, dict_heavy)[0]
                for k, v in instance.items()}, dict_heavy
    elif instance is None:
        return instance, dict_heavy
    # decode fields to json format
    elif type(instance) in [np.ndarray]:
        # todo check if identifier is already in dictionary
        # identifier = '{}'.format(abs(hash(instance.data.tobytes())))
        identifier = 'id_{}'.format(if_key_exists_increment_id(dict_heavy, name_instance, id=1))
        dict_heavy[identifier] = instance
        return identifier, dict_heavy
    elif hasattr(instance, 'dtype'):
        # https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
        return dataclass_to_dicts(instance.item(), dict_heavy)
    elif type(instance) in built_in_types:
        return instance, dict_heavy
    elif isinstance(instance, Enum):
        return dataclass_to_dicts(instance.name, dict_heavy)
    else:
        raise ValueError('Case not implemented')


def _create_instance_from_type_str(type_str, **kwargs):
    # https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module
    module, cls_name = type_str.split('|')
    module_ = importlib.import_module(module)
    class_ = getattr(module_, cls_name)
    return class_(**kwargs)


def create_instance_from_data_dict(type_instance,
                                   data_dict):
    if is_dataclass(type_instance):
        if data_dict is None:
            return None
        kwargs = {}
        _fields = [(data_dict[_field.name], _field) for _field in fields(type_instance) if _field.name in data_dict]
        for _value, _field in _fields:
            if isinstance(_field.type, str):
                _field.type = get_type_hints(type_instance)[_field.name]
            if _field.init is False:
                pass
            elif _field.metadata.get(EXCLUDE_KEY, False):
                kwargs[_field.name] = None
            elif _value is None:
                kwargs[_field.name] = None
            else:
                kwargs[_field.name] = create_instance_from_data_dict(_field.type, _value)
        if 'type' in data_dict:
            return _create_instance_from_type_str(data_dict['type'], **kwargs)
        else:
            return type_instance(**kwargs)
    elif hasattr(type_instance, '__origin__'):
        # https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic
        if type_instance.__origin__ == tuple:
            return tuple([create_instance_from_data_dict(type_instance.__args__[0], item)
                          for item in data_dict])
        if type_instance.__origin__ == list:
            return [create_instance_from_data_dict(type_instance.__args__[0], item)
                    for item in data_dict]
        elif type_instance.__origin__ == dict:
            return {key: create_instance_from_data_dict(type_instance.__args__[0], item)
                    for key, item in data_dict.items()}
        if type_instance.__origin__ == Union:
            for arg in type_instance.__args__:
                if data_dict['type'] in built_in_types_names:
                    return create_instance_from_data_dict(arg, data_dict['value'])
                elif data_dict['type'] in str(f'{arg.__module__}|{arg.__name__}'):
                    return create_instance_from_data_dict(arg, data_dict)
            raise ValueError('specific type not found in given Union type')
    elif type_instance in [np.ndarray, np.array]:
        return data_dict
    elif type_instance in built_in_types:
        return data_dict
    elif isinstance(type_instance, type(Enum)):
        return type_instance[data_dict]
    else:
        raise NotImplementedError()


# import h5py
# def numpy_dict_to_hdf5_file(file, dict_heavy):
#     with h5py.File(file + '.h5', 'w') as data_file:
#         # data_file.create_dataset(name='light', data=json_light)
#         g = data_file.create_group('heavy')
#         for key in dict_heavy:
#             if dict_heavy[key] is not None:
#                 g.create_dataset(name=key, data=dict_heavy[key], compression='gzip', compression_opts=9)
#
#
# def numpy_dict_from_hdf5_file(file):
#     with h5py.File(file + '.h5', 'r') as data_file:
#         dict_heavy = {key: data_file['heavy'][key][:] for key in list(data_file['heavy'].keys())}
#     return dict_heavy


@dataclass
class NumpyJson:
    data: list
    dtype: str

    @classmethod
    def from_array(self, array: np.ndarray):
        return NumpyJson(list(array.tolist()), str(array.dtype))


def numpy_dict_to_zip_file(file, dict_heavy):
    """
    https://stackoverflow.com/questions/40886234/how-to-directly-add-file-to-zip-in-python
    :param file:
    :param dict_heavy:
    :return:
    """
    with zipfile.ZipFile(file + '.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped_f:
        for key, value in dict_heavy.items():
            if isinstance(value, np.ndarray):
                container = NumpyJson(dtype=str(value.dtype),
                                      data=value.tolist())
                zipped_f.writestr(key + '.json', json.dumps(container.__dict__, indent=2))


def numpy_dict_from_zip_file(file):
    with zipfile.ZipFile(file + '.zip', 'r') as zipped_f:
        dict_numpy_json = {item.filename.replace('.json', ''): NumpyJson(**json.loads(zipped_f.read(item.filename),
                                                                                      object_hook=my_decoder))
                           for item in zipped_f.filelist}
        dict_heavy = {key: np.array(item.data, dtype=item.dtype) for key, item in dict_numpy_json.items()}
    return dict_heavy


# @dataclass
# class MetaDataSettings:
#     PRIVATE: Dict[str] = field(default_factory=lambda :{'private': True})# this value is not getting stored or loaded to disk
#

T = TypeVar('T')


def store_files_to_zip(file_path_zip, dict_files: Dict[str, str]):
    logging.info('write: ' + str(file_path_zip))
    if not file_path_zip.parent.exists():
        Path.mkdir(file_path_zip.parent, parents=True)
        print('Created directory ' + str(file_path_zip.parent))
    if file_path_zip.suffix != '.zip':
        file_path_with_dot_zip = file_path_zip.with_suffix(file_path_zip.suffix + '.zip')
    else:
        file_path_with_dot_zip = file_path_zip
    with zipfile.ZipFile(file_path_with_dot_zip, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=2) as zipped_f:
        for key, value in dict_files.items():
            zipped_f.writestr(key, value)


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {'_cplx_': str(obj)}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def my_decoder(dct):
    if '_cplx_' in dct:
        return complex(dct['_cplx_'])
    elif 'data' and 'dtype' in dct:
        return np.array(dct['data'], dtype=dct['dtype'])
    else:
        return dct


def add_suffix(file: Path, suffix: str):
    """
    When file name is ..../file_paremeter=0.1, then pathlib would recognize .1 as the file type.

    However, .1 is not the file type in our application.

    Therefore, this function catches the above case in order to include .1 still in the file name with added suffix.

    :param file:
    :param suffix:
    :return:
    """
    return file.with_name(f'{file.name}{suffix}')


def create_json_from_instance(instance):
    dict_light, dict_heavy = dataclass_to_dicts(instance)
    json_light = json.dumps(dict_light, indent=2, cls=MyJsonEncoder)
    replacements = {'"{}"'.format(key): json.dumps(NumpyJson.from_array(array=value).__dict__, cls=MyJsonEncoder)
                    for key, value in dict_heavy.items()}
    json_light = replace_all(json_light, replacements)
    return json_light


def store(instance, file=None):
    file = Path(file)
    json_light = create_json_from_instance(instance)
    dict_files = {file.name + '.json': json_light}
    store_files_to_zip(file, dict_files)


def load(file):
    file = Path(file)
    with zipfile.ZipFile(add_suffix(file, '.zip'), 'r') as zipped_f:
        dict_data = [json.loads(zipped_f.read(item.filename), object_hook=my_decoder) for item in zipped_f.filelist]
    return dict_data[0]


def _replace_not_excluded_fields(old, new):
    for key, val in old.__dataclass_fields__.items():
        if is_dataclass(old.__getattribute__(key)):
            if not val.metadata.get(EXCLUDE_KEY, False):
                _replace_not_excluded_fields(old.__getattribute__(key), new.__getattribute__(key))
        else:
            if not val.metadata.get(EXCLUDE_KEY, False):
                old.__setattr__(key, new.__getattribute__(key))


class Mode(Enum):  # output file format
    ZIP = 'zip'
    JSON = 'json'


class Persistent:  # on difference between persitable and persistent: https://wikidiff.com/persistent/persistable
    @staticmethod
    def _deal_with_file(file: Union[Path, str]) -> Path:
        if not isinstance(file, Path):
            file = Path(file)
        return file

    @staticmethod
    def _extract_file_name(file: Path) -> str:
        return file.name  # file.stem

    def to_json(self) -> str:
        json_light = create_json_from_instance(self)
        return json_light

    def _store_to_disk_compressed_including_single_json_file(self, file):
        # todo remove uncompressed file if exists
        file = self._deal_with_file(file)
        json_light = self.to_json()
        dict_files = {self._extract_file_name(file) + '.json': json_light}
        store_files_to_zip(file, dict_files)

    # todo
    # def store_to_disk_compressed_including_separate_json_files(self, file):
    #     dict_light, dict_heavy = create_light_and_heavy_part_from_instance(self)
    #     json_light = json.dumps(dict_light, indent=2)
    #
    #     dict_files = {key: json.dumps(value, indent=2) for key, value in dict_heavy}
    #     dict_files[self._extract_file_name(file)] = json_light
    #     store_files_to_zip(file, dict_files)

    def _store_to_disk_uncompressed_single_json_file(self, file):
        json_light = self.to_json()
        write_string_to_file(json_light, add_suffix(file, '.json'))

    def store(self, file, mode: Mode = Mode.ZIP):
        if mode is Mode.ZIP:
            self._store_to_disk_compressed_including_single_json_file(file)
        elif mode is Mode.JSON:
            self._store_to_disk_uncompressed_single_json_file(file)
        else:
            raise NotImplementedError()

    @classmethod
    def _load_from_disk_uncompressed(cls, file: Path) -> T:
        with open(str(add_suffix(file, '.json')), 'r') as data_file:
            json_string = '\n'.join(data_file.readlines())
            dict_data = json.loads(json_string, object_hook=my_decoder)
        return create_instance_from_data_dict(cls, dict_data)

    @classmethod
    def _load_from_disk_compressed(cls, file) -> T:
        dict_data = load(file)
        return create_instance_from_data_dict(cls, dict_data)

    @classmethod
    def load(cls: Type[T], file) -> T:
        file = cls._deal_with_file(file)
        # remove suffix if exists
        if file.suffix in ['.json', '.zip']:
            file = Path(str(file).replace(file.suffix, ''))
        if Path(str(file) + '.json').exists():  # uncompressed case
            return cls._load_from_disk_uncompressed(file)
        elif Path(str(file) + '.zip').exists():  # compressed case
            return cls._load_from_disk_compressed(file)
        else:
            raise FileNotFoundError('No file with supported type found')

    @staticmethod
    def load_as_dict(file):
        return load(file)

    @classmethod
    def from_json(cls, json_: str) -> T:
        return create_instance_from_data_dict(cls, json.loads(json_))

    def update(self, file):
        """
        Updates all fields of an dataclass instance with data from disk, expcept fields mith metadata={'exclude': True}.
        Those excluded fields are not updated.
        :param file:
        :return:
        """
        loaded = self.load(file)
        _replace_not_excluded_fields(self, loaded)
