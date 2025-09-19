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

from drugforge.ml.cli import ml  # noqa: F401, E402, F811
cli.add_command(ml)


from drugforge.dataviz.cli import visualization  # noqa: F401, E402, F811

cli.add_command(visualization)

from drugforge.simulation.cli import simulation  # noqa: F401, E402, F811

cli.add_command(simulation)


from drugforge.data.cli.cli import data  # noqa: F401, E402, F811

cli.add_command(data)
