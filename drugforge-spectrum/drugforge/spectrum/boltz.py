"""
drugforge/spectrum/boltz.py
============================
Pydantic models and pure functions for building Boltz-1 YAML input files.

Boltz-1 takes a single YAML per folding job that describes all chains
(proteins, ligands, nucleic acids). This module covers the single-protein
and protein+ligand cases relevant to the 2A-protease panel.

Public API
----------
BoltzProteinChain   – a protein chain entry in a Boltz YAML
BoltzLigandChain    – a small-molecule ligand entry in a Boltz YAML
BoltzInput          – top-level Boltz YAML input (Pydantic model)
make_boltz_inputs   – FASTA → list of BoltzInput objects (one per sequence)
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from drugforge.spectrum.schema import SequenceList

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BoltzProteinChain(BaseModel):
    """A single protein chain in a Boltz YAML input."""

    id: str = Field("A", description="Chain ID letter.")
    sequence: str = Field(..., description="One-letter amino acid sequence.")

    def to_boltz_dict(self) -> dict:
        return {"protein": {"id": self.id, "sequence": self.sequence}}


class BoltzLigandChain(BaseModel):
    """A small-molecule ligand chain in a Boltz YAML input."""

    id: str = Field("L", description="Chain ID letter for the ligand.")
    smiles: str = Field(..., description="SMILES string for the ligand.")

    def to_boltz_dict(self) -> dict:
        return {"ligand": {"id": self.id, "smiles": self.smiles}}


class BoltzInput(BaseModel):
    """Top-level Boltz YAML input for a single folding job."""

    name: str = Field(..., description="Job name; used to name the output YAML file.")
    version: int = Field(1, description="Boltz input format version.")
    protein_chains: list[BoltzProteinChain] = Field(
        ..., description="Protein chains to include in the folding job."
    )
    ligand_chains: list[BoltzLigandChain] = Field(
        default_factory=list,
        description="Optional small-molecule ligand chains.",
    )

    def to_boltz_dict(self) -> dict:
        """Serialise to the Boltz YAML structure."""
        sequences = [chain.to_boltz_dict() for chain in self.protein_chains]
        sequences += [chain.to_boltz_dict() for chain in self.ligand_chains]
        return {"version": self.version, "sequences": sequences}

    def write(self, output_dir: str | Path) -> Path:
        """Write this input to ``<output_dir>/<name>.yaml`` and return the path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.name}.yaml"
        with open(out_path, "w") as fh:
            yaml.dump(
                self.to_boltz_dict(), fh, default_flow_style=False, sort_keys=False
            )
        return out_path


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def make_boltz_inputs(
    fasta_path: str | Path,
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "L",
) -> list[BoltzInput]:
    """Build Boltz YAML inputs from a FASTA file.

    Creates one :class:`BoltzInput` per sequence. If *ligand_smiles* is
    provided, a ligand chain is appended to every input — useful when folding
    a panel of proteins all against the same small molecule.

    Parameters
    ----------
    fasta_path:
        Path to a FASTA file containing protein sequences.
    ligand_smiles:
        Optional SMILES string for a ligand to include in every input.
    ligand_id:
        Chain ID for the ligand, by default ``"L"``.

    Returns
    -------
    list[BoltzInput]
        One :class:`BoltzInput` per sequence in the FASTA.
    """
    seq_list = SequenceList.from_fasta(fasta_path, aligned=False)

    ligand_chains = (
        [BoltzLigandChain(id=ligand_id, smiles=ligand_smiles)] if ligand_smiles else []
    )

    return [
        BoltzInput(
            name=seq.seq_id,
            protein_chains=[BoltzProteinChain(id="A", sequence=seq.sequence)],
            ligand_chains=ligand_chains,
        )
        for seq in seq_list
    ]
