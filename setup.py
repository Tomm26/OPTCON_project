from setuptools import setup, find_packages

setup(
    name="optcon-project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'matplotlib'
    ]
)