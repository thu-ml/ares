from setuptools import setup, find_packages

setup(
    name='feature_scatter',
    version='0.0.1',
    install_requires=[
        'torch',
        'tqdm',
        'scipy',
        'torchvision',
    ],
    packages=find_packages(),
)
