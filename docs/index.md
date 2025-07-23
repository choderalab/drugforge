Welcome to drugforge's documentation!
=========================================

All pandemics are global health threats. Our best defense is a healthy global antiviral discovery community with a robust pipeline of open discovery tools.
[The AI-driven Structure-enabled Antiviral Platform](https://asapdiscovery.org) is making this a reality!

The toolkit in this repo is a batteries-included drug discovery pipeline being actively developed in a transparent open-source way, with a focus on computational chemistry and informatics support for medicinal chemistry.
Coupled with ASAP's [active data disclosures](https://asapdiscovery.org/outputs) our campaign to develop a new series of antivirals can provide insight into the drug discovery process that is normally conducted behind closed doors.

```{note}
drugforge is not designed as a one size fits all solution for drug discovery, rather it is our opinionated tooling designed with our targets in mind. Your mileage will vary if not using an ASAP target.
```

The `drugforge` toolkit is focused around the following core competencies, organised into submodules:

 - `drugforge-alchemy`: Free energy calculations using [OpenFE](https://openfree.energy/) and [Alchemiscale](https://github.com/openforcefield/alchemiscale)

 - `drugforge-cli`: Command line tools uniting the whole repo.

 - `drugforge-data`: Core data models and integrations with services such as [postera.ai](https://postera.ai/) , [Collaborative Drug Discovery Vault](https://www.collaborativedrug.com/)`, and Diamond Light Source's [Fragalysis](https://fragalysis.diamond.ac.uk/viewer/react/landing) database

 - `drugforge-dataviz`: Data and structure visualization using [3DMol](https://3dmol.csb.pitt.edu) and [PyMOL](https://pymol.org/)

 - `drugforge-docking`: Docking and compound screening with the [OpenEye toolkit](https://docs.eyesopen.com/toolkits/python/index.html)

 - `drugforge-spectrum`: Working with sequence and fitness information and conducting protein structure inference

 - `drugforge-ml`: Structure and graph based ML models for predicting compound activity and other endpoints

 - `drugforge-modelling`: Structure prep and standardisation

 - `drugforge-simulation`: MD simulations and analysis using [OpenMM](https://openmm.org/)

 - `drugforge-workflows`: Workflows that combine components to enable end to end project support




The `drugforge` toolkit is currently focused on [ASAP's targets](https://asapdiscovery.org/pipeline/), which are enumerated on the ASAP website.

See `future` for more information on our roadmap and future plans for the toolkit.


Disclaimer
----------
drugforge is pre-alpha and is under very active development, we make no guarantees around correctness and the API is liable to change rapidly at any time.


```{toctree}
:maxdepth: 5
:hidden:
installation
tutorials/index
guides/index
ecosystem/index
future/index
API/index
```
