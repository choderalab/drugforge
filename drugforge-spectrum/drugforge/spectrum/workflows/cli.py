from pathlib import Path

import click
from drugforge.spectrum.alphafold import (
    make_fold_inputs,
    make_msa_inputs,
    select_best_af3,
)
from drugforge.spectrum.boltz import make_boltz_inputs
from drugforge.spectrum.calculate_rmsd import save_alignment_pymol
from drugforge.spectrum.schema import (
    SequenceList,
    find_bsite_resids,
    run_multiple_sequence_alignment,
    view_alignment,
)
from drugforge.data.util.logging import FileLogger


def _parse_seeds(ctx, param, value: str) -> list[int]:
    """Click callback – convert a comma-separated string to a list of ints."""
    try:
        return [int(s.strip()) for s in value.split(",")]
    except ValueError:
        raise click.BadParameter(
            f"Seeds must be comma-separated integers, got: {value!r}"
        )


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


@cli.command(
    "msa-input", help="Generate AF3 JSON inputs for the MSA (data pipeline) step."
)
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--seeds",
    "-s",
    default="1,2,5,10",
    show_default=True,
    callback=_parse_seeds,
    is_eager=True,
    help="Comma-separated integer model seeds.",
)
@click.option(
    "--description-prefix",
    default="",
    show_default=False,
    help="Optional string prepended to each chain description, e.g. '2A protease'.",
)
def msa_input(fasta, output_dir, seeds, description_prefix):
    """Generate one AF3 JSON per sequence in FASTA for the MSA step.

    MSA fields are left null so AlphaFold 3 runs its data pipeline
    (Jackhmmer / Nhmmer). Run the resulting JSONs with --norun_inference.

    FASTA: path to the input FASTA file.

    OUTPUT_DIR: directory to write per-sequence JSON files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="msa_input", path=output_dir, logfile="msa_input.log"
    ).getLogger()
    inputs = make_msa_inputs(
        fasta_path=fasta, seeds=seeds, description_prefix=description_prefix
    )
    for af3_input in inputs:
        out_path = af3_input.write(output_dir)
        logger.info(f"Wrote {out_path}")
        click.echo(f"  Wrote {out_path}")
    click.secho(f"\nWrote {len(inputs)} MSA-input JSON(s) to {output_dir}", fg="green")


@cli.command("fold-input", help="Generate AF3 fold-input JSONs from MSA outputs.")
@click.argument("msa_output_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--seeds",
    "-s",
    default="1,2,5,10",
    show_default=True,
    callback=_parse_seeds,
    is_eager=True,
    help="Comma-separated integer model seeds.",
)
@click.option(
    "--fasta",
    "-f",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Optional FASTA to control which sequences are processed and their "
        "order. If omitted, every sub-directory in MSA_OUTPUT_DIR is used."
    ),
)
def fold_input(msa_output_dir, output_dir, seeds, fasta):
    """Generate fold-ready AF3 JSONs from pre-computed MSA outputs.

    Reads each <name>/<name>_data.json written by the AF3 data pipeline and
    embeds unpairedMsa + templates for GPU inference with --norun_data_pipeline.

    MSA_OUTPUT_DIR: directory containing AF3 MSA outputs (one sub-dir per sequence).

    OUTPUT_DIR: directory to write per-sequence fold-input JSON files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="fold_input", path=output_dir, logfile="fold_input.log"
    ).getLogger()
    inputs = make_fold_inputs(
        msa_output_dir=msa_output_dir, seeds=seeds, fasta_path=fasta
    )
    for af3_input in inputs:
        out_path = af3_input.write(output_dir)
        logger.info(f"Wrote {out_path}")
        click.echo(f"  Wrote {out_path}")
    click.secho(f"\nWrote {len(inputs)} fold-input JSON(s) to {output_dir}", fg="green")


@cli.command("af3-struct-alignment")
@click.argument(
    "struct_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("ref_pdb", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--chain",
    "-c",
    default="A",
    show_default=True,
    help="Chain ID to use for structural alignment.",
)
@click.option(
    "--pymol-save",
    default="af3_aligned.pse",
    show_default=True,
    help="Filename for the saved PyMOL session (written inside OUTPUT_DIR).",
)
@click.option(
    "--color-by-rmsd",
    is_flag=True,
    default=False,
    help="Color aligned structures by per-residue RMSD in the PyMOL session.",
)
@click.option(
    "--fasta",
    "-f",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional FASTA to control which sequences are processed and their "
        "order. If omitted, every sub-directory in STRUCT_DIR is used."
    ),
)
def af3_struct_alignment(
    struct_dir, ref_pdb, output_dir, chain, pymol_save, color_by_rmsd, fasta
):
    """Align AF3 fold outputs to a reference structure and save a PyMOL session.

    Walks STRUCT_DIR (one sub-directory per sequence), picks the top-ranked
    AF3 CIF model for each sequence, aligns it to REF_PDB, saves each aligned
    structure as a PDB in OUTPUT_DIR, then saves a combined PyMOL session.

    STRUCT_DIR:  root AF3 fold output directory (one sub-dir per sequence).

    REF_PDB:     reference PDB to align all structures against.

    OUTPUT_DIR:  directory to write aligned PDBs and the PyMOL session.
    """
    from drugforge.spectrum.schema import SequenceList

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="af3_struct_alignment",
        path=output_dir,
        logfile="af3_struct_alignment.log",
    ).getLogger()

    # Determine which sequences to process and in what order
    if fasta is not None:
        seq_list = SequenceList.from_fasta(fasta, aligned=False)
        names = [seq.seq_id for seq in seq_list]
    else:
        names = sorted(p.name for p in struct_dir.iterdir() if p.is_dir())

    if not names:
        raise click.ClickException(f"No sequence directories found in {struct_dir}")

    aligned_pdbs = []
    seq_labels = []

    for name in names:
        final_pdb = output_dir / f"{name}_aligned.pdb"
        try:
            rmsd, aligned_pdb = select_best_af3(
                af3_output_dir=struct_dir,
                seq_name=name,
                ref_pdb=ref_pdb,
                chain=chain,
                final_pdb=final_pdb,
            )
            aligned_pdbs.append(aligned_pdb)
            seq_labels.append(name)
            logger.info(f"{name}: RMSD = {rmsd:.3f} Å  →  {aligned_pdb}")
            click.echo(f"  {name}: RMSD = {rmsd:.3f} Å")
        except FileNotFoundError as e:
            logger.warning(str(e))
            click.secho(f"  WARN: {e}", fg="yellow")

    if not aligned_pdbs:
        raise click.ClickException("No structures were successfully aligned.")

    session_save = output_dir / pymol_save
    save_alignment_pymol(
        aligned_pdbs, seq_labels, str(ref_pdb), str(session_save), chain, color_by_rmsd
    )
    logger.info(f"Saved PyMOL session to {session_save}")
    click.secho(
        f"\nAligned {len(aligned_pdbs)} structure(s). PyMOL session → {session_save}",
        fg="green",
    )


@cli.command("make-boltz-input", help="Generate Boltz YAML inputs from a FASTA file.")
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--ligand-smiles",
    "-l",
    default=None,
    help="SMILES string for a ligand to include in every input YAML.",
)
@click.option(
    "--ligand-id",
    default="L",
    show_default=True,
    help="Chain ID to assign to the ligand.",
)
def make_boltz_input(fasta, output_dir, ligand_smiles, ligand_id):
    """Generate one Boltz YAML per sequence in FASTA.

    Boltz uses --use_msa_server so no separate MSA step is required.
    Run the resulting YAMLs with: boltz predict <yaml> --use_msa_server

    FASTA:      path to the input FASTA file.

    OUTPUT_DIR: directory to write per-sequence YAML files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(
        logname="make_boltz_input", path=output_dir, logfile="make_boltz_input.log"
    ).getLogger()
    inputs = make_boltz_inputs(
        fasta_path=fasta,
        ligand_smiles=ligand_smiles,
        ligand_id=ligand_id,
    )
    for boltz_input in inputs:
        out_path = boltz_input.write(output_dir)
        logger.info(f"Wrote {out_path}")
        click.echo(f"  Wrote {out_path}")
    click.secho(f"\nWrote {len(inputs)} Boltz YAML(s) to {output_dir}", fg="green")


if __name__ == "__main__":
    cli()
