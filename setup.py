from setuptools import setup, find_packages
import shutil
import os
import sys
import call_ai_grapher

def delete_path_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

delete_path_if_exists("build")
delete_path_if_exists("call_ai_grapher.egg-info")
delete_path_if_exists("dist")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
  name='call_ai_grapher',
  version="0.0.1",
  description="Improve handwriting using GANS",
  long_description=long_description,
  packages=find_packages(
      where='.',
      include=['call_ai_grapher*']
  ),
  package_data={'':['*']},
  include_package_data=True,
  entry_points={
    'console_scripts': [
        'run=call_ai_grapher.__main__:main',
    ],
    # 'group_1': 'run=call_ai_grapher.__main__:main'
  },
  install_requires=[
    'setuptools'
  ]
)