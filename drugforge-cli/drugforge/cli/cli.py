import warnings

import click

# Check for Openeye and suggest to install it if not
try:
    import openeye.oechem as oechem  # noqa: F401
except ImportError:
    warnings.warn(
        "Cannot import openeye oechem. OpenEye toolkits are required for this software to work, please install them using `mamba install -c openeye openeye-toolkits`"
    )


@click.group()
def cli(help="Command-line interface for drugforge"): ...


from drugforge.workflows.docking_workflows.cli import docking  # noqa: F401, E402, F811

cli.add_command(docking)

from drugforge.workflows.prep_workflows.cli import (  # noqa: F401, E402, F811
    protein_prep,
)

cli.add_command(protein_prep)

from drugforge.alchemy.cli.cli import alchemy  # noqa: F401, E402, F811

cli.add_command(alchemy)

from drugforge.workflows.spectrum_workflows.cli import (  # noqa: F401, E402, F811
    spectrum,
)

cli.add_command(spectrum)

# we do not have the ML package on macos, causing issues if we try to import it
# so we import inside a try except to allow the rest of the CLI to work
try:
    from drugforge.ml.cli import ml  # noqa: F401, E402, F811

    cli.add_command(ml)
except ImportError:
    print("ML package not available, skipping ML CLI command.")

from drugforge.dataviz.cli import visualization  # noqa: F401, E402, F811

cli.add_command(visualization)

from drugforge.simulation.cli import simulation  # noqa: F401, E402, F811

cli.add_command(simulation)


from drugforge.data.cli.cli import data  # noqa: F401, E402, F811

cli.add_command(data)
