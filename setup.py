"""
Helper functions to find word trends.
https://github.com/adbar/shoten
"""


import re

from pathlib import Path
from setuptools import setup, find_packages


def get_version(package):
    "Return package version as listed in `__version__` in `init.py`"
    package_metadata = Path(package, '__init__.py').read_text()
    return re.search('__version__ = [\'"]([^\'"]+)[\'"]', package_metadata)[1]


def get_long_description():
    "Return the README"
    with open("README.md", "r", encoding="utf-8") as filehandle:
        long_description = filehandle.read()
    return long_description


#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = [
                   'courlan >= 1.1.0',
                   'lxml >= 5.2.2',
                   'numpy',
                   'scikit-learn >= 0.24.2',
                   'simplemma == 0.9.0',
               ]

setup_requirements = []

test_requirements = ['pytest>=3', 'pytest-cov']

setup(
    author="Adrien Barbaresi",
    author_email='barbaresi@bbaw.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        #'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    description="...",
    entry_points = {
        'console_scripts': ['shoten=shoten.cli:main'],
    },
    install_requires=requirements,
    license="GPLv3+",
    long_description= get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=['nlp'],
    name='shoten',
    package_data={},
    packages=find_packages(include=['shoten', 'shoten.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/adbar/shoten',
    version=get_version('shoten'),
    zip_safe=False,
)
