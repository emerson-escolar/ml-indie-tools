import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-small-datasets-domschl",
    version="0.0.1",
    author="Dominik Schlösser",
    author_email="dsc@dosc.net",
    description="A collection of small datasets for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/domschl/ml-small-datasets",
    project_urls={
        "Bug Tracker": "https://github.com/domschl/ml-small-datasets/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
