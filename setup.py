from setuptools import setup
import os
from glob import glob

package_name = 'thermo'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
     (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='Naveed Riaziat',
 maintainer_email='nriaziat@jhu.edu',
 description='TODO: Package description',
 license='TODO: License declaration',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'traj_node = thermo.traj_node:main',
             'controller_node = thermo.controller_node:main',
     ],
   },
)