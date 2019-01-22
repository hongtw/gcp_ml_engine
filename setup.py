from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'scikit-learn==0.19.1', 
    "requests >= 2.18.0",
    'joblib==0.11',
    'numpy==1.14.5', 
    'scipy==1.1.0'
    ]

setup(
    name='trainer',
    version='0.5',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.',
)