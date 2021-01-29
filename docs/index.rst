.. PsychRNN documentation master file, created by
   sphinx-quickstart on Mon May 11 12:57:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PsychRNN's documentation!
====================================


.. image:: https://github.com/murraylab/PsychRNN/workflows/tests/badge.svg?branch=v{release}
    :target: https://github.com/murraylab/PsychRNN/actions?query=workflow%3Atests
    :alt: Build Status

.. image:: https://codecov.io/gh/murraylab/PsychRNN/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/murraylab/PsychRNN
    :alt: codecov

This package is intended to help cognitive scientists easily translate task designs from human or primate behavioral experiments into a form capable of being used as training data for a recurrent neural network.

We have isolated the front-end task design, in which users can intuitively describe the conditional logic of their task from the backend where gradient descent based optimization occurs. This is intended to facilitate researchers who might otherwise not have an easy implementation available to design and test hypothesis regarding the behavior of recurrent neural networks in different task environements.

Start with `Hello World <notebooks/Minimal_Example.ipynb>`_ to get a quick sense of what PsychRNN does. Then go through the `Simple Example <notebooks/PerceptualDiscrimination.ipynb>`_ to get a feel for how to customize PsychRNN. The rest of `Getting Started <quickstart.rst>`_ will help guide you through using available features, defining your own task, and even defining your own model.

Release announcments are posted on the `psychrnn mailing list <https://www.freelists.org/list/psychrnn>`_ and on `GitHub <https://github.com/murraylab/PsychRNN>`_.

Code is written and upkept by: `Daniel B. Ehrlich <https://github.com/dbehrlich>`_, `Jasmine T. Stone <https://github.com/syncrostone/>`_, `David Brandfonbrener <https://github.com/davidbrandfonbrener>`_, and `Alex Atanasov <https://github.com/ABAtanasov>`_.

Contact: psychrnn@gmail.com

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installing
   apidoc/index
   quickstart
   reference
