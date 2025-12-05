import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

with open('dev_requirements.txt') as open_file:
    install_requires_dev = open_file.read()

setuptools.setup(
    name='ttavlm',
    version='1.0.0',
    url='',
    packages=setuptools.find_packages(),
    author='Anonymous',
    author_email='',
    description='Test-Time Adaptation Library for VLMs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={
        'dev': install_requires_dev,
    },
)
