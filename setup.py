from setuptools import setup, find_packages
from ares import __version__


def fetch_requirements(filename):
    requirements = []
    with open(filename) as f:
        for ln in f.read().split("\n"):
            ln = ln.strip()
            if '--index-url' in ln:
                ln = ln.split('--index-url')[0].strip()
            requirements.append(ln)
        return requirements

setup(
    name="ares_pytorch",
    version=__version__,
    author="Xiao Yang, Chuang Liu, Chang Liu",
    description="ARES 2.0 - A pytorch library for adversarial robustness",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Adversarial Robustness, Deep Learning, Library, PyTorch",
    packages=find_packages(),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.10.0",
)