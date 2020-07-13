#!/usr/bin/env python
#-*- coding:utf-8 -*-

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

from setuptools import setup, find_packages

setup(
    name = "imap",
    version = "0.1.0",
    keywords = ("single-cell RNA-sequencing technologies", "neural network", "GAN", "batch removal"),
    description = "The integration of single-cell RNA-sequencing datasets from multiple sources is critical for deciphering cell-cell heterogeneities and interactions in complex biological systems. We present a novel unsupervised batch removal framework, called iMAP, based on two state-of-art deep generative models – autoencoders and generative adversarial networks.",
    long_description = long_description,
    long_description_content_type='text/markdown',
    license = "MIT Licence",

    url = "https://github.com/Svvord/",
    author = "Dongfang Wang, Siyu Hou",
    author_email = "housy17@mails.tsinghua.edu.cn",
    maintainer = "Siyu Hou",
    maintainer_email = "housy17@mails.tsinghua.edu.cn",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = [
        "numpy",
        "scanpy",
        "torch",
        "pandas",
        "annoy",
        ]
)
