.. PsychRNN documentation master file, created by
   sphinx-quickstart on Mon May 11 12:57:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PsychRNN's documentation!
====================================


[Travis and Codecov badges redacted for double-blind review]

This package is intended to help cognitive scientists easily translate task designs from human or primate behavioral experiments into a form capable of being used as training data for a recurrent neural network.

We have isolated the front-end task design, in which users can intuitively describe the conditional logic of their task from the backend where gradient descent based optimization occurs. This is intended to facilitate researchers who might otherwise not have an easy implementation available to design and test hypothesis regarding the behavior of recurrent neural networks in different task environements.

Start with `Hello World <notebooks/Minimal_Example.ipynb>`_ to get a quick sense of what PsychRNN does. Then go through the `Simple Example <notebooks/PerceptualDiscrimination.ipynb>`_ to get a feel for how to customize PsychRNN. The rest of `Getting Started <quickstart.rst>`_ will help guide you through using available features, defining your own task, and even defining your own model.

Release announcments are posted on the psychrnn mailing list and on GitHub [Links redacted for double-blind review].

Code is written and upkept by: [Authors redacted for double-blind review].

Contact: psychrnn@gmail.com

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installing
   apidoc/index
   quickstart
