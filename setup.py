import json
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent.resolve()
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "bioimageio" / "core" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text())["version"]


_ = setup(
    name="bioimageio.core",
    version=VERSION,
    description="Python functionality for the bioimage model zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/core-bioimage-io-python",
    author="Bioimage Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    packages=find_namespace_packages(exclude=["tests"]),
    install_requires=[
        "bioimageio.spec ==0.5.3.6",
        "h5py",
        "imagecodecs",
        "imageio>=2.10",
        "loguru",
        "numpy",
        "pydantic-settings>=2.5,<3",
        "pydantic>=2.7.0,<3",
        "requests",
        "ruyaml",
        "tqdm",
        "typing-extensions",
        "xarray",
    ],
    include_package_data=True,
    extras_require={
        "pytorch": (pytorch_deps := ["torch>=1.6,<3", "torchvision", "keras>=3.0,<4"]),
        "tensorflow": ["tensorflow", "keras>=2.15,<4"],
        "onnx": ["onnxruntime"],
        "dev": (
            pytorch_deps
            + [
                "black",
                # "crick",  # currently requires python<=3.9
                "jupyter",
                "jupyter-black",
                "matplotlib",
                "onnx",
                "onnxruntime",
                "packaging>=17.0",
                "pre-commit",
                "pdoc",
                "pyright==1.1.394",
                "pytest-cov",
                "pytest",
            ]
        ),
    },
    project_urls={
        "Bug Reports": "https://github.com/bioimage-io/core-bioimage-io-python/issues",
        "Source": "https://github.com/bioimage-io/core-bioimage-io-python",
    },
    entry_points={"console_scripts": ["bioimageio = bioimageio.core.__main__:main"]},
)
