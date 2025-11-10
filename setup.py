from setuptools import setup, find_packages

setup(
    name="anemonefish_acoustics",
    version="0.4.0",
    description="Machine learning tools for identifying and classifying anemonefish sounds",
    author="Lucia Yllan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
) 