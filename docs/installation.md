Installation
===============

This page details how to get started with `drugforge` and how to install it on your system.

There are three ways to install `drugforge`:

1. From conda-forge (recommended)
2. Use the provided Docker image
3. Developer installation from source

Installation from conda-forge
-----------------------------

The easiest way to install `drugforge` is to use the mamba (or conda) package manager. You can install `drugforge` from the `conda-forge` channel using the following command:
The openeye package is not available in the conda-forge channel, so you need to install it from the openeye channel. You will need to have an OpenEye license to use some functionality in the package.
You can request a free academic license from the [OpenEye website](https://docs.eyesopen.com/toolkits/python/index.html).

```bash
mamba create -n drugforge python=3.10
mamba activate drugforge
mamba install -c conda-forge -c openeye drugforge openeye-toolkits
```

This would install the whole suite of packages available in `drugforge`. In case you only want one or a subset of the subpackages, you can always use `drugforge-<subpkg>` instead. For example, installing alchemy and docking would be

```bash
mamba create -n drugforge python=3.10
mamba activate drugforge-sub
mamba install -c conda-forge -c openeye drugforge-alchemy drugforge-docking openeye-toolkits
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

To install `drugforge` from source, you will need to clone the repository, create a compatible base environment with mamba (or conda), we recommend using only the dependencies for `drugforge` for this step.
Install the development dependencies/utils (such as openeye-toolkits, pytest, ipython, etc.).
And finally, install all the subpackages with `pip` using an editable install (so changes get automatically updated). This needs a for loop and compatibility config mode.
You can do this using the following commands:

```bash
git clone git@github.com:choderalab/drugforge.git
cd drugforge
mamba create --name drugforge-dev --only-deps drugforge-*  # Install ONLY dependencies of drugforge and subpkgs
mamba activate drugforge-dev
mamba install -c conda-forge -c openeye pytest-xdist ipython notebook openeye-toolkits
for DIR in $( ls -d drugforge-*/ ); do pip install --no-deps -e $DIR --config-settings editable_mode=compat; done
```

Please note that this can take several minutes (~20 mins in a modern machine), solving the environments is a costly step.

This will install the package in editable mode, so you can make changes to the code and see the changes reflected in the package.
