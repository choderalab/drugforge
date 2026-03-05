"""
drugforge/spectrum/alphafold.py
================================
Pydantic models and pure functions for building AlphaFold 3 JSON input files.

Public API
----------
Af3ProteinChain   – AF3 JSON representation of a single protein chain
Af3Input          – top-level AF3 JSON input (Pydantic model)
make_msa_inputs   – FASTA → list of MSA-stage Af3Input objects
make_fold_inputs  – MSA output dir → list of fold-stage Af3Input objects
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from drugforge.spectrum.schema import ProteinSequence, SequenceList

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Af3ProteinChain(BaseModel):
    """AF3 JSON representation of a single protein chain."""

    id: str = Field("A", description="Chain ID letter used in the AF3 JSON.")
    sequence: str = Field(..., description="One-letter amino acid sequence.")
    description: Optional[str] = Field(None, description="Human-readable label.")
    unpairedMsa: Optional[str] = Field(
        None,
        description=(
            "Pre-computed unpaired MSA in A3M format. "
            "Set to None to let AF3 build the MSA (data pipeline). "
            "Set to '' to run MSA-free."
        ),
    )
    pairedMsa: Optional[str] = Field(
        None,
        description=(
            "Pre-computed paired MSA in A3M format. "
            "Set to None alongside unpairedMsa=None so AF3 builds both. "
            "Set to '' when providing a custom unpairedMsa for a single chain."
        ),
    )
    templates: Optional[list] = Field(
        None,
        description=(
            "List of structural templates. None → AF3 searches for templates. "
            "[] → run template-free."
        ),
    )

    def to_af3_dict(self, version: int = 2) -> dict:
        """Serialise to the AF3 JSON 'protein' sub-dict."""
        d: dict = {"id": self.id, "sequence": self.sequence}
        if self.description is not None:
            if version >= 4:
                d["description"] = self.description
        # Always write MSA fields explicitly so AF3 interprets them correctly
        d["unpairedMsa"] = self.unpairedMsa
        d["pairedMsa"] = self.pairedMsa
        d["templates"] = self.templates
        return d


class Af3Input(BaseModel):
    """Top-level AF3 JSON input for a single folding job."""

    name: str = Field(..., description="Job name; used to name output files.")
    model_seeds: list[int] = Field(
        default_factory=lambda: [1, 2, 5, 10],
        description="List of integer random seeds. At least one required.",
    )
    chains: list[Af3ProteinChain] = Field(
        ..., description="Protein chains to include in the folding job."
    )
    dialect: str = Field("alphafold3", description="Must be 'alphafold3'.")
    version: int = Field(2, description="AF3 JSON format version.")

    def to_af3_dict(self) -> dict:
        """Serialise to the full AF3 JSON structure."""
        return {
            "name": self.name,
            "modelSeeds": self.model_seeds,
            "sequences": [{"protein": chain.to_af3_dict()} for chain in self.chains],
            "dialect": self.dialect,
            "version": self.version,
        }

    def write(self, output_dir: str | Path) -> Path:
        """Write this input to ``<output_dir>/<name>.json`` and return the path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.name}.json"
        with open(out_path, "w") as fh:
            json.dump(self.to_af3_dict(), fh, indent=2)
        return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_msa_output(msa_output_dir: Path, name: str) -> tuple[str, list, str]:
    """Extract ``(unpairedMsa, templates, sequence)`` from an AF3 MSA output.

    AF3 writes ``<msa_output_dir>/<name>/<name>_data.json`` after the data
    pipeline completes.

    Parameters
    ----------
    msa_output_dir:
        Root directory of AF3 MSA outputs.
    name:
        Sequence / job name.

    Returns
    -------
    tuple[str, list, str]
        ``(unpairedMsa, templates, sequence)``
    """
    data_json = msa_output_dir / name / f"{name}_data.json"
    if not data_json.exists():
        raise FileNotFoundError(
            f"MSA output not found for '{name}': expected {data_json}"
        )
    with open(data_json) as fh:
        data = json.load(fh)
    protein = data["sequences"][0]["protein"]
    return protein["unpairedMsa"], protein["templates"], protein["sequence"]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def make_msa_inputs(
    fasta_path: str | Path,
    seeds: list[int] | None = None,
    description_prefix: str = "",
) -> list[Af3Input]:
    """Build MSA-stage AF3 inputs from a FASTA file.

    Each returned :class:`Af3Input` has all MSA fields set to ``None`` so that
    AlphaFold 3 runs its data pipeline (Jackhmmer / Nhmmer) to build MSAs.
    Run the resulting JSONs with ``--norun_inference``.

    Parameters
    ----------
    fasta_path:
        Path to a FASTA file containing protein sequences.
    seeds:
        Integer model seeds. Defaults to ``[1, 2, 5, 10]``.
    description_prefix:
        Optional string prepended to each chain description, e.g. ``"2A protease"``.

    Returns
    -------
    list[Af3Input]
        One :class:`Af3Input` per sequence in the FASTA.
    """
    if seeds is None:
        seeds = [1, 2, 5, 10]

    seq_list = SequenceList.from_fasta(fasta_path, aligned=False)

    return [
        Af3Input(
            name=seq.seq_id,
            model_seeds=seeds,
            chains=[
                Af3ProteinChain(
                    id="A",
                    sequence=seq.sequence,
                    description=(
                        f"{description_prefix} – {seq.seq_id}"
                        if description_prefix
                        else seq.seq_id
                    ),
                    unpairedMsa=None,
                    pairedMsa=None,
                    templates=None,
                )
            ],
        )
        for seq in seq_list
    ]


def make_fold_inputs(
    msa_output_dir: str | Path,
    seeds: list[int] | None = None,
    fasta_path: str | Path | None = None,
) -> list[Af3Input]:
    """Build fold-stage AF3 inputs from pre-computed MSA outputs.

    Reads each ``<name>/<name>_data.json`` written by the AF3 data pipeline
    and embeds the ``unpairedMsa`` and ``templates`` into a new
    :class:`Af3Input` ready for GPU inference. Run with
    ``--norun_data_pipeline``.

    Parameters
    ----------
    msa_output_dir:
        Directory containing one sub-directory per sequence, each holding the
        AF3 data-pipeline output JSON.
    seeds:
        Integer model seeds. Defaults to ``[1, 2, 5, 10]``.
    fasta_path:
        Optional FASTA used to control which sequences are processed and in
        what order. When omitted every sub-directory in *msa_output_dir* is
        used (sorted alphabetically).

    Returns
    -------
    list[Af3Input]
        One :class:`Af3Input` per sequence.
    """
    if seeds is None:
        seeds = [1, 2, 5, 10]

    msa_dir = Path(msa_output_dir)

    if fasta_path is not None:
        seq_list = SequenceList.from_fasta(fasta_path, aligned=False)
        names = [seq.seq_id for seq in seq_list]
    else:
        names = sorted(p.name for p in msa_dir.iterdir() if p.is_dir())

    if not names:
        raise ValueError(f"No sequence directories found in {msa_dir}")

    inputs: list[Af3Input] = []
    for name in names:
        unpaired_msa, templates, sequence = _read_msa_output(msa_dir, name)
        inputs.append(
            Af3Input(
                name=name,
                model_seeds=seeds,
                chains=[
                    Af3ProteinChain(
                        id="A",
                        sequence=sequence,
                        description=name,
                        unpairedMsa=unpaired_msa,
                        pairedMsa="",  # single chain – no inter-chain pairing
                        templates=templates,
                    )
                ],
            )
        )
    return inputs
