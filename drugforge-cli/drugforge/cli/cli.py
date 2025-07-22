import click


@click.group()
def cli(help="Command-line interface for drugforge"): ...


from drugforge.workflows.docking_workflows.cli import (  # noqa: F401, E402, F811
    docking,
)

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

# TODO: Re-enable ML CLI when ready
# "ML CLI not available. Please install drugforge[ml] to use the ML CLI commands."
# The ML subpackage requires a refactor to work with Pydantic 2.
# This has been done in PR: https://github.com/choderalab/asapdiscovery/pull/2
# However it results in breaking changes to other parts of the repo that need to be fixed in a future release.
# This also highlights a current issue with our cli organization, in which the cli for any package depends on being able
# to import all of the other packages.
# We will also address this in a future release.

# from drugforge.ml.cli import ml  # noqa: F401, E402, F811
# cli.add_command(ml)


from drugforge.dataviz.cli import visualization  # noqa: F401, E402, F811

cli.add_command(visualization)

from drugforge.simulation.cli import simulation  # noqa: F401, E402, F811

cli.add_command(simulation)


from drugforge.data.cli.cli import data  # noqa: F401, E402, F811

cli.add_command(data)
