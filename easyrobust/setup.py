from setuptools import setup, find_packages
import os
import shutil

project_name = 'easyrobust'

#install the package
setup(
  name=project_name,
  description='EasyRobust: a tool for training robust image models',
  author='xiaofeng',
  version='0.1.0',
  packages=find_packages(),
  install_requires=['timm==0.4.12', 'torch', 'torchvision', 'imagecorruptions', 'numpy', 'einops'],
)

# #clean up after build
# for dir_name in ['build',project_name+'.egg-info','__pycache__','dist']:
#   if os.path.exists(dir_name):
#     shutil.rmtree(dir_name)