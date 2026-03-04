from Bio import SeqIO
from drugforge.spectrum.schema import (
    ProteinSequence,
    SequenceList,
    run_multiple_sequence_alignment,
)
import pytest


def test_protein_sequence():
    seq = ProteinSequence(
        id="P12345", aligned=False, sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"
    )
    assert seq.id == "P12345"
    assert seq.sequence == "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"


@pytest.fixture()
def unaligned_fasta_file(fasta_alignment_path, tmpdir):
    sequences = SequenceList.from_fasta(fasta_alignment_path, aligned=True)
    seq_list = SequenceList(
        aligned=False,
        sequences=[sequence.get_unaligned_sequence() for sequence in sequences],
    )
    return seq_list.to_fasta(tmpdir / "unaligned.fasta")


class TestSequenceList:
    def test_from_aligned_fasta(self, fasta_alignment_path):
        sequences = SequenceList.from_fasta(fasta_alignment_path, aligned=True)

        with pytest.raises(ValueError):
            sequences = SequenceList.from_fasta(fasta_alignment_path, aligned=False)

    def test_from_fasta_unaligned(self, unaligned_fasta_file):
        sequences = SequenceList.from_fasta(unaligned_fasta_file, aligned=False)

        with pytest.raises(ValueError):
            sequences = SequenceList.from_fasta(unaligned_fasta_file, aligned=True)

    def test_fasta_roundtrip_unaligned(self, unaligned_fasta_file, tmpdir):
        sequences = SequenceList.from_fasta(unaligned_fasta_file, aligned=False)
        created_path = sequences.to_fasta(tmpdir / "unaligned.fasta")
        roundtripped = SequenceList.from_fasta(created_path, aligned=False)
        assert sequences == roundtripped

    def test_csv_roundtrip_unaligned(self, unaligned_fasta_file, tmpdir):
        sequences = SequenceList.from_fasta(unaligned_fasta_file, aligned=False)
        created_path = sequences.to_fasta(tmpdir / "unaligned.fasta")
        roundtripped = SequenceList.from_fasta(created_path, aligned=False)
        assert sequences == roundtripped


class TestSequenceAlignment:
    def test_alignment(self, fasta_alignment_path, unaligned_fasta_file):
        unaligned_sequences = SequenceList.from_fasta(
            unaligned_fasta_file, aligned=False
        )

        aligned_sequences = run_multiple_sequence_alignment(unaligned_sequences)

        reference_aligned_sequences = SequenceList.from_fasta(
            fasta_alignment_path, aligned=True
        )

        assert aligned_sequences == reference_aligned_sequences
