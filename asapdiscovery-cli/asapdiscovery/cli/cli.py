import click


@click.group()
def cli(help="Command-line interface for asapdiscovery"): ...


from asapdiscovery.workflows.docking_workflows.cli import (  # noqa: F401, E402, F811
    docking,
)

cli.add_command(docking)

from asapdiscovery.workflows.prep_workflows.cli import (  # noqa: F401, E402, F811
    protein_prep,
)

cli.add_command(protein_prep)

from asapdiscovery.alchemy.cli.cli import alchemy  # noqa: F401, E402, F811

cli.add_command(alchemy)

# TODO: Re-enable ML CLI when ready
# "ML CLI not available. Please install asapdiscovery[ml] to use the ML CLI commands."
# The ML subpackage requires a refactor to work with Pydantic 2.
# This has been done in PR: https://github.com/choderalab/asapdiscovery/pull/2
# However it results in breaking changes to other parts of the repo that need to be fixed in a future release.
# This also highlights a current issue with our cli organization, in which the cli for any package depends on being able
# to import all of the other packages.
# We will also address this in a future release.

# from asapdiscovery.ml.cli import ml  # noqa: F401, E402, F811
# cli.add_command(ml)


from asapdiscovery.spectrum.cli import spectrum  # noqa: F401, E402, F811

cli.add_command(spectrum)


from asapdiscovery.dataviz.cli import visualization  # noqa: F401, E402, F811

cli.add_command(visualization)

from asapdiscovery.simulation.cli import simulation  # noqa: F401, E402, F811

cli.add_command(simulation)


from asapdiscovery.data.cli.cli import data  # noqa: F401, E402, F811

cli.add_command(data)
