from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy==1.14.5', 
    'scikit-learn==0.19.1', 
    'joblib==0.11',
    # 'scipy==1.1.0'
    ]

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.',
)