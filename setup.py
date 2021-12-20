#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    history = history_file.read()

requirements = [
                   'numpy',
                   'scikit-learn >= 0.24',
                   'simplemma >= 0.5.0',
                   'htmldate >= 1.0.0',
                   'courlan >= 0.6.0',
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
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    description="...",
    entry_points = {
        'console_scripts': ['shoten=shoten.cli:main'],
    },
    install_requires=requirements,
    license="GPLv3+",
    long_description=readme, # + '\n\n' + history,
    include_package_data=True,
    keywords=['nlp'],
    name='shoten',
    package_data={},
    packages=find_packages(include=['shoten', 'shoten.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/adbar/shoten',
    version='0.1.0',
    zip_safe=False,
)
