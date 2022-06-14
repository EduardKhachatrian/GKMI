from setuptools import setup, find_packages

"""

requirements = [
"numpy>=1.15.4",
"matplotlib>=2.2.4",
"scikit-learn>=0.20.2",
"scipy>=1.2.1",
]
"""

setup(
description="GKMI - Unsupervised Information Selection Method",
version="0.1",
url = 'https://github.com/EduardKhachatrian/GKMI',
author='EdKh',
author_email='khachatrianeduard@gmail.com',
license='MIT',
#install_requires=requirements,
include_package_data=True,
keywords="gkmi",
name="gkmi",

packages=[
'src',
],

test_suite="tests",
zip_safe=False,
)