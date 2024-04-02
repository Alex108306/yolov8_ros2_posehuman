from setuptools import find_packages, setup

package_name = 'cam_detect_human'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='giang.nht108201@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_detect = cam_detect_human.yolov8_node:main',
            'yolov8_3d = cam_detect_human.detect3d_node:main',
            'debug_yolov8 = cam_detect_human.debug_node:main',
            "yolov8_tracking = cam_detect_human.tracking_node:main"
        ],
    },
)
