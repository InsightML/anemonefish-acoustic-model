from setuptools import setup, find_packages

setup(
    name="anemonefish_acoustics",
    version="0.1.0",
    description="Machine learning tools for identifying and classifying anemonefish sounds",
    author="Lucia Yllan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "torchaudio",
        "scikit-learn",
        "librosa",
        "soundfile",
        "jupyter",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
            "sphinx-autobuild",
        ],
    },
    python_requires=">=3.8",
) 