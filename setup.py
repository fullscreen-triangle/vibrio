#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vibrio",
    version="0.1.0",
    author="Kundai Farai Sachikonye",
    author_email="kundai.sachikonye@bitspark.com",
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
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "diffusers>=0.16.0",
        "accelerate>=0.20.0",
        "faiss-cpu>=1.7.4",
        "huggingface-hub>=0.16.4",
        "timm>=0.9.2",
        "einops>=0.6.1",
        "safetensors>=0.3.2",
        "peft>=0.5.0",
        "bitsandbytes>=0.41.0",
        "pydub>=0.25.1",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "ffmpeg-python>=0.2.0",
        "pillow>=10.0.0",
        "tokenizers>=0.13.3",
        "datasets>=2.12.0",
        "imageio>=2.31.1",
        "pyarrow>=12.0.0", 
        "scikit-learn>=1.3.0",
        "onnxruntime>=1.15.0",
        "optimum>=1.12.0",
    ],
    entry_points={
        "console_scripts": [
            "vibrio=main:main",
            "vibrio-calibrate=calibrate:main",
        ],
    },
) 