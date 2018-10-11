import setuptools
from distutils.core import setup

setup(
    name='PsychRNN',
    version='0.9dev',
    packages=['psychrnn', 'psychrnn.tasks', 'psychrnn.backend', 'psychrnn.backend.models'],
    license='MIT',
    long_description=open('README.md').read(),

    author="Alex Atanasov, David Brandfonbrener, Daniel Ehrlich, Jasmine Stone",
    author_email="daniel.ehrlich@yale.edu",
    description="Easy-to-use package for the modeling and analysis of neural network dynamics, directed towards cognitive neuroscientists.",
    keywords="neuroscience, modeling, analysis, neural networks",
    url="https://github.com/dbehrlich/PsychRNN",

)