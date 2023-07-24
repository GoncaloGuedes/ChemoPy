from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

# Read requirements.txt and extract dependencies
with open("requirements.txt", "r") as req_file:
    requirements = req_file.read()

setup(
    name="chemopy",
    version="0.0.10",
    description="An package that contains preprocessing, model transfer and orthogonal corrections for Spectroscopy",
    packages=find_packages(where="."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gonçalo Guedes",
    author_email="goncalo.manuelguedes@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,  # Use the extracted dependencies from requirements.txt
)
