import setuptools
from distutils.core import setup

with open("psychrnn/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()

setup(

    name='PsychRNN',
    version=__version__,
    packages=['psychrnn', 'psychrnn.tasks', 'psychrnn.backend', 'psychrnn.backend.models'],
    license='MIT',
    long_description=long_desc,
    long_description_content_type='text/markdown',

    author_email="psychrnn@gmail.com",
    description="Easy-to-use package for the modeling and analysis of neural network dynamics, directed towards cognitive neuroscientists.",
    keywords="neuroscience, modeling, analysis, neural networks",
    url="URL redacted for double-blind review",
    project_urls={
    'Documentation': 'readthedocs URL redacted for double-blind review',
    'Mailing List': 'URL redacted for double-blind review',
    },

    install_requires=['python_version>="2.7",!="3.0*",!="3.1*",!="3.2*",!="3.3*"', 'tensorflow']

)