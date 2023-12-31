
import pathlib

import pkg_resources
import setuptools

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semshap",
    version="0.0.1",
    author="Michele Cafagna",
    author_email = "michele.cafagna@um.edu.mt",
    description="Explain VL generative models using sentence-based explanation and visual semantic priors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michelecafagna26/vl-shap",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: license.txt",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
)