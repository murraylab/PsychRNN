Installation Guide
==================

System requirements
-------------------

- python = 2.7 or 3.6
- `numpy <http://www.numpy.org/>`_
- `tensorflow <https://www.tensorflow.org/>`_ >= 1.14

- For notebook demos, `jupyter <https://jupyter.org/>`_
- For notebook demos, `ipython <https://ipython.org/>`_
- For plotting features, `matplotlib <https://matplotlib.org/>`_

PsychRNN was developed to work with both python 2.7 and 3.6 using tensorflow 1.14 and 2.0. General compatibility is expected for python 2 and 3 and tensorflow 1.14, 1.15 and 2.

Installation
------------

Normally, you can install with: ::

	pip install --index-url https://test.pypi.org/simple/ psychrnn=1.0.0-alpha

Alternatively, you can clone and install the source files via the `GitHub repository <https://github.com/murraylab/psychrnn/blob/|release|>`_:

.. code-block:: bash

        git clone https://github.com/murraylab/PsychRNN.git
        cd PsychRNN
        python setup.py install

Contributing
------------

Please report bugs to https://github.com/murraylab/psychrnn/issues.  This
includes any problems with the documentation.  Fixes (in the form of
pull requests) for bugs are greatly appreciated.

Feature requests are welcome but may or may not be accepted due to limited
resources. If you implement the feature yourself we are open
to accepting it in PsychRNN.  If you implement a new feature in PsychRNN,
please do the following before submitting a pull request on Github:

- Make sure your code is clean and well commented
- If appropriate, update the official documentation in the ``docs/``
  directory
- Write unit tests and optionally integration tests for your new
  feature in the ``tests/`` folder.
- Ensure all existing tests pass (``pytest`` returns without
  error)

For all other questions or comments, contact psychrnn@gmail.com.