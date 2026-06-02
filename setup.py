# setup.py
from setuptools import setup, find_packages

setup(
    name='qkd_routing',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'torch', 'gymnasium',
        'matplotlib', 'pyyaml', 'pandas'
    ]
)
