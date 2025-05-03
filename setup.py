#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vibrio",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Human speed analysis framework using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vibrio",
    py_modules=["main", "calibrate"],
    packages=["modules"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "ultralytics>=8.0.0",
        "filterpy>=1.4.5",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "vibrio=main:main",
            "vibrio-calibrate=calibrate:main",
        ],
    },
) 