from setuptools import setup, find_packages

setup(
    name="deeplure_saree_rec",
    version="1.0.0",
    author="DeepLure AI Engineering Candidate",
    description="Color-Invariant Saree Design & Motif Recognition System using Metric Learning",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md", "r", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.7",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pillow>=9.5.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
