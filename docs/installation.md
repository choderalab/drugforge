Installation
===============

This page details how to get started with `drugforge` and how to install it on your system.

There are three ways to install `drugforge`:

1. From conda-forge (recommended)
2. Use the provided Docker image
3. Developer installation from source

Installation from conda-forge (WIP)
-----------------------------------

The easiest way to install `drugforge` is to use the mamba (or conda) package manager. You can install `drugforge` from the `conda-forge` channel using the following command:
The openeye package is not available in the conda-forge channel, so you need to install it from the openeye channel. You will need to have an OpenEye license to use some functionality in the package.
You can request a free academic license from the [OpenEye website](https://docs.eyesopen.com/toolkits/python/index.html).

```bash
mamba create -n drugforge python=3.10
mamba activate drugforge
mamba install -c conda-forge drugforge
mamba install -c openeye openeye-toolkits

```

Installation from Docker (WIP)
------------------------------

Currently there is not docker image published for this package.

A Docker image is available for `drugforge` on the Github container registry [ghcr.io](https://github.com/choderalab/drugforge/pkgs/container/drugforge) You can pull the image using the following command:

```bash
 docker pull ghcr.io/choderalab/drugforge:main
```

Now you can run the image using the following command:

```bash
docker run -it ghcr.io/choderalab/drugforge:main
```

Note that the Docker image assumes that your OpenEye license is located at `~/.OpenEye/oe_license.txt`. If your license is located elsewhere, you can mount it to the container using the `-v` flag and the relevant environment variables.


Developer installation from source
----------------------------------

To install `drugforge` from source, you will need to clone the repository, setup a compatible environment with mamba (or conda) and install the package using `pip`. You can do this using the following commands:

```bash
git clone git@github.com:choderalab/drugforge.git
cd drugforge
mamba env create -f devtools/conda-envs/drugforge-ubuntu-latest.yml # chose relevant file for your platform
mamba activate drugforge
cp devtools/repo_installer.sh . && chmod +x repo_installer.sh && ./repo_installer.sh
```

This will install the package in editable mode, so you can make changes to the code and see the changes reflected in the package. You can also run the tests using the following command:
