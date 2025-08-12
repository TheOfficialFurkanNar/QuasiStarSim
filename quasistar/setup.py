from setuptools import setup, find_packages

# Uzun açıklama için genellikle README.md okunur
with open("../README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pioneerai",
    version="0.1.0",
    author="Furkan",
    author_email="pioneerai.code@gmail.com",
    description="Sistemsel analiz odaklı AI framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheOfficialFurkanNar/QuasiStarSim.git",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "click",
        "pyyaml",
        "h5py; extra == 'hdf5'"
    ],
    extras_require={
        "dev": ["pytest", "pytest-mpl", "flake8", "black"]
    },
    entry_points={
        "console_scripts": [
            "pioneerai = pioneerai.__main__:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
