from setuptools import setup, find_packages

requirements = [
    "scipy",
    "scikit-image",
    "scikit-learn",
    "matplotlib",
]
setup(
description="GKMI - Unsupervised Information Selection Method",
version="0.1",
url = 'https://github.com/EduardKhachatrian/GKMI',
author='EdKh',
author_email='khachatrianeduard@gmail.com',
license='MIT',
install_requires=requirements,
include_package_data=True,
keywords="gkmi",
name="gkmi",
packages=[
'src',
],
test_suite="tests",
zip_safe=False,
)