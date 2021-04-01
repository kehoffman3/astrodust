"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='astrodust',
    version='1.0.0',
    description='A library for predicting the distribution of dust particles in protoplanetary disks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kehoffman3/astrodust',
    author='UVA Astronomy Capstone Group',
    author_email='keh4nb@virginia.edu',
   classifiers=[  
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='astronomy,machine learning,planets,random forest',
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=[
            'joblib>=1.0',
            'xgboost>=1.2',
            'tqdm>=4.50',
            'scikit-learn>=0.24.0',
        ], 
    options={"bdist_wheel": {"universal": True}}
)