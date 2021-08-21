dataclass-persistence
==========================

This program can be used to make dataclasses persistent by adding store and load functionality.
The dataclass is stored in .json format which is by default compressed inside of a .zip file.

What makes dataclass-persistence special?
   * Support for numpy arrays
   * Support for nested dataclasses
   * Human readable storage format with small file size


Usage
-----
Let your dataclass inherit from :code:`Persistent`.
Then the dataclass can be stored on disk using :code:`.store()` and loaded from disk using
:code:`.load()`.

In the example below, we create an instance of dataclass, which is stored to and loaded from disk.

.. code-block:: python

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

On disk the code above produces `my_file.zip` which contains `my_file.json`:

.. code-block:: json

    {
      "parameter_a": "my_string",
      "array": {"data": [0, 0], "dtype": "int32"}
    }