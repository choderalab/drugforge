import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from drugforge.data.schema.schema_base import DataModelAbstractBase
from typing_extensions import Self
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from warnings import warn


class ProteinSequence(BaseModel):
    seq_id: str = Field(..., description="Unique identifier for the protein sequence")
    aligned: bool = Field(
        ...,
        description="Indicates whether the sequence is aligned (True) or unaligned (False)",
    )
    sequence: str = Field(
        ...,
        description="Amino acid sequence of the protein",
    )

    @model_validator(mode="after")
    def validate_sequence(self) -> Self:
        if not self.aligned:
            valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
            if not all(residue in valid_amino_acids for residue in self.sequence):
                raise ValueError(
                    "Protein sequence contains invalid characters. Only standard amino acids are allowed."
                )
        return self

    def get_unaligned_sequence(self) -> Self:
        if self.aligned:
            return ProteinSequence(
                seq_id=self.seq_id,
                sequence=self.sequence.replace("-", ""),
                aligned=False,
            )
        else:
            return self


class SequenceList(BaseModel):
    aligned: bool = Field(
        None,
        description="Indicates whether all the sequences are aligned (True) or unaligned (False).",
    )
    sequences: list[ProteinSequence] = Field(
        ..., description="List of protein sequences"
    )

    def __iter__(self):
        return iter(self.sequences)

    @model_validator(mode="after")
    def validate_sequence_length(self) -> Self:
        if self.aligned:
            seq_lengths = {len(seq.sequence) for seq in self.sequences}
            if len(seq_lengths) > 1:
                raise ValueError(
                    "All sequences must be the same length when 'aligned' is True."
                )
        return self

    @classmethod
    def from_fasta(cls, input_fasta, aligned: bool):
        """Load sequences from a FASTA file and return a list of ProteinSequence or AlignedSequence objects."""
        input_fasta = Path(input_fasta)
        if not input_fasta.exists():
            raise ValueError(f"FASTA file does not exist: {input_fasta}")
        if not input_fasta.suffix == ".fasta":
            raise ValueError("Fasta file must be in FASTA format")
        sequences = []
        for record in SeqIO.parse(input_fasta, "fasta"):
            sequences.append(
                ProteinSequence(
                    seq_id=record.id, sequence=str(record.seq), aligned=aligned
                )
            )
        return cls(aligned=aligned, sequences=sequences)

    def to_bio_seq_records(self) -> list[SeqRecord]:
        seq_recs = [
            SeqRecord(Seq(sequence.sequence), id=sequence.seq_id)
            for sequence in self.sequences
        ]
        return seq_recs

    def to_dataframe(self) -> pd.DataFrame:
        records = [seq.model_dump() for seq in self.sequences]
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        records = df.to_dict(orient="records", index=True)
        sequences = [ProteinSequence(**record) for record in records]
        aligned_list = [seq.aligned for seq in sequences]
        if not all(aligned_list):
            if any(aligned_list):
                warn(f"Some, but not all of the sequences are aligned: {sequences}")
            aligned = False
        if all(aligned_list):
            aligned = True
        return cls(aligned=aligned, sequences=sequences)

    def to_fasta(self, output_fasta: str | Path):
        """Convert sequences to a FASTA file"""
        output_fasta = Path(output_fasta)
        output_fasta.parent.mkdir(parents=True, exist_ok=True)
        if not output_fasta.suffix == ".fasta":
            raise ValueError(
                "Fasta file must be in FASTA format and have the .fasta extension"
            )
        seq_recs = self.to_bio_seq_records()
        SeqIO.write(seq_recs, output_fasta, "fasta")

        if not output_fasta.exists():
            raise RuntimeError(
                f"SeqIO failed to write these sequences to {output_fasta}:\n {seq_recs}"
            )

        return output_fasta

    def to_csv(self, output_csv: str | Path):
        """Convert sequences to a CSV file"""
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if not output_csv.suffix == ".csv":
            raise ValueError("CSV file must be in .csv format")
        df = self.to_dataframe()
        df.to_csv(output_csv, index=False)
        if not output_csv.exists():
            raise RuntimeError(
                f"CSV failed to write these sequences to {output_csv}:\n {df}"
            )
        return output_csv

    @classmethod
    def from_csv(cls, input_csv: str | Path) -> Self:
        """Load sequences from a CSV file"""
        input_csv = Path(input_csv)
        if not input_csv.suffix == ".csv":
            raise ValueError("CSV file must be in .csv format")
        df = pd.read_csv(input_csv)
        return cls.from_dataframe(df)


def run_multiple_sequence_alignment(sequences: SequenceList) -> SequenceList:
    """Run a multiple sequence alignment using MAFFT.

    Writes the input sequences to a temporary FASTA file, runs MAFFT,
    captures the aligned output in a second temporary file, and returns
    the result as an aligned SequenceList.

    Parameters
    ----------
    sequences : SequenceList
        Unaligned sequences to be aligned.

    Returns
    -------
    SequenceList
        A new SequenceList with ``aligned=True`` containing the MAFFT-aligned sequences.
    """
    with (
        tempfile.NamedTemporaryFile(
            suffix=".fasta", mode="w", delete=False
        ) as input_file,
        tempfile.NamedTemporaryFile(
            suffix=".fasta", mode="w", delete=False
        ) as output_file,
    ):
        input_path = Path(input_file.name)
        output_path = Path(output_file.name)

    try:
        SeqIO.write(sequences.to_bio_seq_records(), input_path, "fasta")
        cmd = ["mafft", str(input_path)]
        with output_path.open("w") as out_fh:
            sp_output = subprocess.run(
                cmd, stdout=out_fh, stderr=subprocess.PIPE, check=True
            )

        return SequenceList.from_fasta(output_path, aligned=True)
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
