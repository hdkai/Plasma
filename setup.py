# 
#   Plasma
#   Copyright (c) 2021 Yusuf Olokoba.
#

from setuptools import find_packages, setup

# Get readme
with open("README.md", "r") as readme:
    long_description = readme.read()

# Get version
with open("plasma/version.py") as version_source:
    gvars = {}
    exec(version_source.read(), gvars)
    version = gvars["__version__"]

# Setup
setup(
    name="torchplasma",
    version=version,
    author="Yusuf Olokoba",
    author_email="hi@hdk.ai",
    description="Differentiable image editing operations for computational photography in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
	python_requires=">=3.6",
    install_requires=[
        "imageio",
        "torch",
        "torchvision"
    ],
    url="https://github.com/hdkai/Plasma",
    packages=find_packages(include=["plasma", "plasma.*"]),
    package_data={ },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ],
)