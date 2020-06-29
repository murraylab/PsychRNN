.. PsychRNN documentation master file, created by
   sphinx-quickstart on Mon May 11 12:57:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PsychRNN's documentation!
====================================


.. image:: https://api.travis-ci.com/murraylab/PsychRNN.svg?branch=|release|
    :target: https://api.travis-ci.com/murraylab/PsychRNN
    :alt: Build Status

.. image:: :codecov:|release|
    :target: https://codecov.io/gh/murraylab/PsychRNN
    :alt: codecov

This package is intended to help cognitive scientist easily translate task designs from human or primate behavioral experiments into a form capable of being used as training data for a recurrent neural network.


We have isolated the front-end task design, in which users can intuitively describe the conditional logic of their task from the backend where gradient descent based optimization occurs. This is intended to facilitate researchers who might otherwise not have an easy implementation available to design and test hypothesis regarding the behavior of recurrent neural networks in different task environements.


Code is written and upkept by: @davidbrandfonbrener @dbehrlich @ABAtanasov @syncrostone

Contact: psychrnn@gmail.com

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installing
   apidoc/index
   quickstart
