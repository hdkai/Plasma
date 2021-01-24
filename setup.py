# 
#   Plasma
#   Copyright (c) 2021 Homedeck, LLC.
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
    name="plasma",
    version=version,
    author="Homedeck, LLC.",
    author_email="info@homedeck.io",
    description="Differentiable image editing framework.",
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
    packages=find_packages(include=["plasma"]),
    package_data={
        "plasma.io": ["data/*.tif"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ],
)