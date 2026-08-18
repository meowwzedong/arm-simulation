from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'arm_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name, 'launch'),glob('launch/*.py')),
        (os.path.join('share',package_name, 'urdf'),glob('urdf/*')),
        (os.path.join('share',package_name, 'config'),glob('config/*')),
        (os.path.join('share',package_name, 'meshes'),glob('meshes/*')),
        (os.path.join('share',package_name, 'models'),glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mahi2',
    maintainer_email='tasinmuhtadi1@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arm_controller = arm_sim.arm_controller:main',
            'arm_controller_ik = arm_sim.arm_controller_ik:main',
            'arm_controller_key_ik = arm_sim.arm_controller_key_ik:main'
        ],
    },
)
