from pathlib import Path

import click
from drugforge.spectrum.schema import (
    SequenceList,
    find_bsite_resids,
    run_multiple_sequence_alignment,
    view_alignment,
)
from drugforge.data.util.logging import FileLogger


@click.group("spectrum-cli")
def cli():
    pass


@cli.command(
    "align-fasta", help="Use mafft to run multiple sequence alignment in FASTA format"
)
@click.argument("input_fasta")
@click.argument("output_dir", type=click.Path())
def align_fasta(input_fasta, output_dir):
    """"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="align_fasta", path=output_dir, logfile="align_fasta.log"
    ).getLogger()
    logger.info(f"Aligning {input_fasta}...")
    seq_list = SequenceList.from_fasta(input_fasta, aligned=False)
    aligned = run_multiple_sequence_alignment(seq_list)
    aligned.serialize(output_dir)
    logger.info(f"Successfully aligned {len(aligned.sequences)} sequences.")


@cli.command(
    "vizualize-alignment", help="Visualize aligned sequences using a bokeh plot."
)
@click.argument("input_fasta")
@click.argument("output_dir", type=click.Path())
@click.option(
    "--color-by-group/--no-color-by-group",
    default=None,
    help=(
        "Color sequences by amino acid group match (--color-by-group) or by amino acid "
        "identity (--no-color-by-group). By default both plots are produced."
    ),
)
@click.option(
    "--pdb",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Path to a PDB file containing a protein-ligand complex. "
        "When provided, binding site residues are highlighted with a blue box overlay."
    ),
)
@click.option(
    "--ligres",
    default="LIG",
    show_default=True,
    help="Residue name of the ligand in the PDB file.",
)
@click.option(
    "--chain",
    default="A",
    show_default=True,
    help="Chain ID of the protein/ligand in the PDB file.",
)
@click.option(
    "--bsite-dist",
    default=4.5,
    show_default=True,
    type=float,
    help="Distance in Å from the ligand used to define the binding site.",
)
def vizualize_alignment(
    input_fasta, output_dir, color_by_group, pdb, ligres, chain, bsite_dist
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="vizualize_alignment",
        path=output_dir,
        logfile="vizualize_alignment.log",
    ).getLogger()

    seq_list = SequenceList.from_fasta(input_fasta, aligned=True)

    bsite_resids = None
    if pdb is not None:
        logger.info(f"Detecting binding site residues from {pdb}...")
        bsite_resids = find_bsite_resids(
            pdb, ligres=ligres, chain=chain, bsite_dist=bsite_dist
        )
        logger.info(
            f"Found {len(bsite_resids)} binding site residues: {bsite_resids.tolist()}"
        )

    # Determine which modes to render: both by default, one if explicitly specified.
    if color_by_group is None:
        modes = [(True, "colored_by_group"), (False, "colored_by_amino_acid")]
    elif color_by_group:
        modes = [(True, "colored_by_group")]
    else:
        modes = [(False, "colored_by_amino_acid")]

    for by_group, file_name in modes:
        view_alignment(
            seq_list,
            color_by_group=by_group,
            start_idx=1,
            output_dir=output_dir,
            file_name=file_name,
            plot_width=2400,
            bsite_resids=bsite_resids,
        )
        logger.info(f"Saved alignment plot to {output_dir / file_name}.html")


if __name__ == "__main__":
    cli()
