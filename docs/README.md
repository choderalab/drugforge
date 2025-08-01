# Compiling drugforge's Documentation

For the docs to compile properly, you will need to have the full drugforge environment installed. 

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To install the needed dependencies, you can run:

```bash
 mamba env update -f docs/requirements.yaml -n YOUR_DRUGFORGE_ENVIRONMENT_NAME 
```

You can then build the docs with:
```bash
sphinx-build -M html DOCS_SOURCE TEST_DIR
```

And examine them with:
```bash
cd TEST_DIR/html
python -m http.server
```

It will give you a local port to view the webpage in your browser.


A configuration file for [Read The Docs](https://readthedocs.org/) (readthedocs.yaml) is included in the top level of the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/ and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the project.

If you would like to use Read The Docs with `autodoc` (included automatically) and your package has dependencies, you will need to include those dependencies in your documentation yaml file (`docs/requirements.yaml`).
