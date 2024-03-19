
import os
from glob import glob
from setuptools import setup, find_packages

exec(open("mapillm/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MAPI_LLM",
    version=__version__,
    description="A Python package for the MAPI_LLM project",
    author="Mayk Caldas",
    author_email="maykcaldas@gmail.edu",
    url="https://github.com/maykcaldas/MAPI_LLM",
    license="MIT",
    packages=['mapillm'],
    install_requires=[
        "numpy",
        "pandas",
        "openai",
        "langchain",
        "langchain_openai",
        "langchainhub",
        "mp_api",
        "request",
        "rdkit",
        "transformers",
        "pymatgen",
        "faiss-cpu",
        "reaction-network",
        "python-dotenv",
        "numexpr",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)