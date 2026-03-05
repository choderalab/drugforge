import io
import subprocess
import tempfile
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, LabelSet, LinearAxis, Range1d
from bokeh.models.glyphs import Rect, Text

# Bokeh imports
from bokeh.plotting import figure, output_file, save
from drugforge.spectrum.seq_alignment import get_colors_by_aa_group, get_colors_protein
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


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

    def serialize(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.to_fasta(output_dir / "sequences.fasta")
        self.to_csv(output_dir / "sequences.csv")

    def to_bio_alignment_obj(
        self,
    ) -> MultipleSeqAlignment:
        """Returns a MultipleSeqAlignment object representing the alignment"""
        buf = io.StringIO()
        SeqIO.write(self.to_bio_seq_records(), buf, "fasta")
        buf.seek(0)
        return AlignIO.read(buf, "fasta")


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
    with tempfile.NamedTemporaryFile(
        suffix=".fasta", mode="w", delete=False
    ) as input_file:
        input_path = Path(input_file.name)

    try:
        SeqIO.write(sequences.to_bio_seq_records(), input_path, "fasta")
        cmd = ["mafft", str(input_path)]
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        aligned_records = list(SeqIO.parse(io.StringIO(result.stdout), "fasta"))
        aligned_sequences = [
            ProteinSequence(seq_id=rec.id, sequence=str(rec.seq), aligned=True)
            for rec in aligned_records
        ]
        return SequenceList(aligned=True, sequences=aligned_sequences)
    finally:
        input_path.unlink(missing_ok=True)


def find_bsite_resids(
    pdb: str | Path,
    ligres: str = "LIG",
    chain: str = "A",
    bsite_dist: float = 4.5,
    res_threshold: int = 5,
) -> np.ndarray:
    """Find binding site residues in a single protein-ligand complex based on ligand proximity.

    Unlike the version in ``calculate_rmsd``, this function operates on a single
    input structure and does not require a separate reference PDB or alignment
    step — the ligand must already be present in ``pdb``.

    Parameters
    ----------
    pdb : str or Path
        Path to the PDB file containing the protein-ligand complex.
    ligres : str, optional
        Residue name of the ligand, by default "LIG"
    chain : str, optional
        Chain ID of the protein and ligand, by default "A"
    bsite_dist : float, optional
        Distance from the ligand in angstroms used to define the binding site,
        by default 4.5
    res_threshold : int, optional
        Minimum residue ID to be considered a binding site residue. Avoids
        assigning terminal residues incorrectly, by default 5

    Returns
    -------
    np.ndarray
        Sorted array of binding site residue IDs.

    Raises
    ------
    ValueError
        If no ligand atoms are found for ``ligres`` in ``chain``.
    ValueError
        If no binding site residues are found above ``res_threshold``.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(pdb))

    lig_atoms = u.select_atoms(f"chainid {chain} and resname {ligres}")
    if len(lig_atoms) == 0:
        raise ValueError(
            f"No ligand atoms found for resname '{ligres}' in chain '{chain}' of {pdb}"
        )

    bs_atoms = u.select_atoms(
        f"protein and chainid {chain} and around {bsite_dist} resname {ligres}"
    )

    bs_resids = np.unique(bs_atoms.resids)
    bs_resids = bs_resids[bs_resids >= res_threshold]

    if len(bs_resids) == 0:
        raise ValueError(
            f"No binding site residues found within {bsite_dist} Å of '{ligres}' "
            f"with residue ID >= {res_threshold}"
        )

    return bs_resids


def view_alignment(
    sequence_list: SequenceList,
    fontsize="11pt",
    plot_width=800,
    file_name="alignment",
    color_by_group=False,
    start_idx=0,
    skip=4,
    max_mismatch=2,
    reorder: list = None,
    output_dir=None,
    scores: list = None,
    x_offsets: float = None,
    bsite_resids: np.ndarray | list | None = None,
):
    """ "Bokeh sequence alignment view
        From: https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner

    Parameters
    ----------
    sequence_list : SequenceList
    fontsize : str, optional
        Size of aminoacid one-letter IDs, by default "11pt"
    plot_width : int, optional
        width of alignment plot, by default 800
    file_name : str, optional
        suffix for html file, by default "alignment"
    color_by_group : bool, optional
        View mode where matching aminoacids are colored, by default False
    start_idx : int, optional
        Index of first aminiacid of reference sequence, by default 0
    skip : int, optional
        Skip for displayed indexes of reference sequence , by default 4
    max_mismatch : int, optional
        How many mismatches are tolerated for highlighted group match, by default 2
    reorder : list, optional
        List of indices to reorder sequences, by default None
    output_dir : str or Path, optional
        Directory to save the output html file, by default None (saves to cwd)
    scores : list, optional
        List of per-sequence scores to display as right-side labels, by default None
    x_offsets : float, optional
        X position of the score labels relative to the alignment width. Defaults
        to ``N + 8`` (just past the right edge) when scores are provided.
    bsite_resids : array-like, optional
        Residue IDs of binding site residues (in the reference sequence, i.e. the
        last sequence in the alignment). Continuous runs of binding site columns
        are highlighted with a blue box overlay on both plots, by default None.

    Returns
    -------
    (bokeh.Column, str)
        Bokeh Column of layouts, path to saved html file.
    """
    if not sequence_list.aligned:
        raise ValueError("Sequence list must have aligned sequences.")

    # The function takes a biopython alignment object as input.
    aln = sequence_list.to_bio_alignment_obj()
    if reorder is not None:
        aln_ref = aln[:1]  # ref
        aln_sorted = [aln[int(i)] for i in reorder]
        aln_ref.extend(aln_sorted)
        aln = aln_ref

    aln = aln[::-1]  # So outputs are ordered from top to bottom
    seqs = [rec.seq for rec in (aln)]  # Each sequence input
    text = [i for s in list(seqs) for i in s]  # Al units joind on same list

    N = len(seqs[-1])
    S = len(seqs)

    # Shorten the description for display — take the part after the last ":"
    # in the description field, falling back to rec.id if no ":" is present.
    def matches(x):
        return x.split(":")[-1] if ":" in x else x

    desc = [
        matches(rec.description) if rec.description != rec.id else rec.id for rec in aln
    ]
    colors_dict = {"exact": "white", "group": "orange", "none": "red"}

    # List with ALL colors
    # By aminoacid group or exact match
    if color_by_group:
        col_colors = []
        font_colors = []
        match_keys = []
        for col in range(N):  # Go through each column
            # Note: AlignIO item retrieval is done through a get_item function, so this has to be done with a loop
            col_string = aln[:, col]
            color, font_color, match_key = get_colors_by_aa_group(
                col_string, max_mismatch, colors_dict
            )
            col_colors.append(color)
            font_colors.append(font_color)
            match_keys.append(match_key)
        colors = col_colors * S
        # Append each font_color list "colum-wise"
        font_colors = np.array(font_colors).T.flatten()
    else:
        colors = get_colors_protein(seqs)
        font_colors = ["black"] * len(colors)

    # Defining x indexes only for non-gap characters of ref sequence (seqs[-1])
    seq_array = np.array(list(seqs[-1]))
    x_non_gap = np.full(len(seqs[-1]), " ", dtype="<U3")
    non_gap_idx = np.where(seq_array != "-")[0]
    current_idx = start_idx
    x_non_gap_locs = []
    # Iterate to indexes (this way we skip the gaps in the middle)
    for idx in non_gap_idx:
        if idx in non_gap_idx[::skip]:  # Skips every given index
            x_non_gap[idx] = str(current_idx)
            x_non_gap_locs.append(idx)
        current_idx += 1

    x = np.arange(0, N)
    y = np.arange(0, S, 1)

    # Map binding site residue IDs onto alignment column indices.
    # non_gap_idx[i] is the alignment column for the (start_idx + i)-th residue
    # of the reference sequence (seqs[-1]). We offset by start_idx to convert
    # residue numbers back to 0-based positions into non_gap_idx.
    bsite_spans = []  # list of (col_start, col_end) alignment column spans
    if bsite_resids is not None:
        bsite_resids = np.asarray(bsite_resids)
        # Convert residue numbers to 0-based positions within non_gap_idx
        bsite_positions = bsite_resids - start_idx
        # Keep only positions that fall within the reference sequence length
        valid = bsite_positions[
            (bsite_positions >= 0) & (bsite_positions < len(non_gap_idx))
        ]
        bsite_cols = np.sort(non_gap_idx[valid])

        # Group consecutive alignment columns into contiguous spans
        if len(bsite_cols) > 0:
            span_start = bsite_cols[0]
            span_end = bsite_cols[0]
            for col in bsite_cols[1:]:
                if col == span_end + 1:
                    span_end = col
                else:
                    bsite_spans.append((span_start, span_end))
                    span_start = span_end = col
            bsite_spans.append((span_start, span_end))
    # creates a 2D grid of coords from the 1D arrays
    xx, yy = np.meshgrid(x, y)
    # flattens the arrays
    gx = xx.ravel()
    gy = yy.flatten()
    # use recty for rect coords with an offset
    recty = gy + 0.5
    # now we can create the ColumnDataSource with all the arrays
    # logging.info(f"Aligning {S} sequences of lenght {N}")
    # ColumnDataSource is a JSON dict that maps names to arrays of values
    source = ColumnDataSource(dict(x=gx, y=gy, recty=recty, text=text, colors=colors))
    plot_height = len(seqs) * 10 + 50
    x_range = Range1d(gx[0] - 1, N + 8, bounds="auto")  # (start, end)
    if N > 150:
        viewlen = 150
    else:
        viewlen = N
    # view_range is for the close up view
    view_range = (gx[0] - 1, viewlen)
    tools = "xpan, xwheel_zoom, reset, save"

    # Custom right-side labels — only rendered when scores are provided
    if scores is not None:
        _x_offsets = x_offsets if x_offsets is not None else N + 8
        right_labels1 = [f"{round(score, 1)}%" for score in scores][::-1]
        source2 = ColumnDataSource(
            data=dict(
                x=[_x_offsets] * len(desc),
                y=desc,
                labels=right_labels1,
            )
        )
        labels = LabelSet(
            x="x",
            y="y",
            text="labels",
            level="glyph",
            x_offset=0,
            y_offset=0,
            source=source2,
            text_align="left",
            text_baseline="middle",
            text_font_size=str(int(fontsize[:-2]) - 2) + "pt",
        )
    else:
        labels = None

    # entire sequence view (no text, with zoom)
    p1 = figure(
        title=None,
        width=plot_width,
        height=plot_height,
        x_range=x_range,
        y_range=desc,
        tools=tools,
        min_border=0,
    )
    p1.toolbar_location = None
    # Rect simply places rectangles of with "width" into the positions defined by x and y
    rects = Rect(
        x="x",
        y="recty",
        width=1,
        height=1,
        fill_color="colors",
        line_color=None,
        fill_alpha=0.6,
    )
    # Source does mapping from keys in rects to values in ColumnDataSource definition
    p1.add_glyph(source, rects)
    p1.grid.visible = False
    p1.xaxis.major_label_text_font_style = "bold"
    p1.yaxis.major_label_text_font_size = "8pt"
    p1.yaxis.minor_tick_line_width = 0
    p1.yaxis.major_tick_line_width = 0
    if labels is not None:
        p1.add_layout(labels)

    def _add_bsite_boxes(plot, spans, n_seqs):
        """Overlay a blue box for each continuous run of binding site columns."""
        if not spans:
            return
        from bokeh.models import BoxAnnotation

        for col_start, col_end in spans:
            box = BoxAnnotation(
                left=col_start - 0.5,
                right=col_end + 0.5,
                fill_color="steelblue",
                fill_alpha=0.25,
                line_color="steelblue",
                line_alpha=0.6,
                line_width=1.5,
            )
            plot.add_layout(box)

    _add_bsite_boxes(p1, bsite_spans, S)

    plot_height = len(seqs) * 20 + 30

    # sequence text view with ability to scroll along x axis
    p2 = figure(
        title=None,
        width=plot_width,
        height=plot_height,
        x_range=view_range,
        y_range=desc,
        tools=tools,
        min_border=0,
        toolbar_location="below",
    )
    # Text does the same thing as rectangles but placing letter (or words) instead, aligned accordingly
    text_source = ColumnDataSource(
        dict(x=gx, y=gy, recty=recty, text=text, colors=font_colors)
    )
    glyph = Text(
        x="x",
        y="y",
        text="text",
        text_color="colors",
        text_align="center",
        text_font_size=fontsize,
    )
    rects = Rect(
        x="x",
        y="recty",
        width=1,
        height=1,
        fill_color="colors",
        line_color=None,
        fill_alpha=0.4,
    )

    # Blank plot to hold the position labels
    p_blank = figure(
        width=plot_width,
        height=40,
        x_range=view_range,
        y_range=Range1d(0, 1),
        title=None,
        toolbar_location=None,
        tools="",
        outline_line_alpha=0,
    )
    p_blank.xaxis.visible = False
    p_blank.yaxis.visible = False
    p_blank.grid.visible = False
    label_source = ColumnDataSource(dict(x=x, y=[0.05] * len(x), text=x_non_gap))
    labels_b = Text(
        x="x",
        y="y",
        text="text",
        text_color="black",
        text_align="center",
        text_font_size=str(int(fontsize[:-2]) - 2) + "pt",
    )
    p2.add_glyph(text_source, glyph)
    p2.add_glyph(source, rects)
    p_blank.add_glyph(label_source, labels_b)
    if labels is not None:
        p2.add_layout(labels)
    _add_bsite_boxes(p2, bsite_spans, S)
    _add_bsite_boxes(p_blank, bsite_spans, S)

    view_range = Range1d(gx[0] - 1, viewlen)
    p2.grid.visible = True
    p2.xaxis.major_label_text_font_style = "bold"
    p2.yaxis.major_label_text_font_style = "bold"
    p2.yaxis.minor_tick_line_width = 0
    p2.yaxis.major_tick_line_width = 0
    p2.xaxis.major_label_text_font_size = "0pt"
    p2.add_layout(
        LinearAxis(major_label_text_font_size="0pt", ticker=list(x_non_gap_locs)),
        "above",
    )
    p2.x_range = view_range
    p_blank.x_range = view_range

    # --- Legend ---
    # Build a row of labelled swatches explaining the colours used in the plot.
    from bokeh.models import Div

    if color_by_group:
        legend_items = [
            ("Exact match (all identical)", "white", "black"),
            ("Group match (same amino acid group)", "orange", "black"),
            ("No match", "red", "black"),
        ]
        legend_title = "Color key: amino acid group matching"
    else:
        # Per-amino-acid colour map from _AMINO_ACID_COLORS
        _AA_COLORS = {
            "A": "red",
            "R": "blue",
            "N": "green",
            "D": "yellow",
            "C": "orange",
            "Q": "purple",
            "E": "cyan",
            "G": "magenta",
            "H": "pink",
            "I": "brown",
            "L": "gray",
            "K": "lime",
            "M": "teal",
            "F": "navy",
            "P": "olive",
            "S": "maroon",
            "T": "silver",
            "W": "gold",
            "Y": "skyblue",
            "V": "violet",
            "-": "white",
        }
        _AA_NAMES = {
            "A": "Ala",
            "R": "Arg",
            "N": "Asn",
            "D": "Asp",
            "C": "Cys",
            "Q": "Gln",
            "E": "Glu",
            "G": "Gly",
            "H": "His",
            "I": "Ile",
            "L": "Leu",
            "K": "Lys",
            "M": "Met",
            "F": "Phe",
            "P": "Pro",
            "S": "Ser",
            "T": "Thr",
            "W": "Trp",
            "Y": "Tyr",
            "V": "Val",
            "-": "Gap",
        }
        legend_items = [
            (f"{aa} ({_AA_NAMES[aa]})", color, "black")
            for aa, color in _AA_COLORS.items()
        ]
        legend_title = "Color key: amino acid identity"

    def _swatch_html(label, fill, text_color="black", border="black"):
        return (
            f'<span style="display:inline-flex;align-items:center;margin:2px 6px;">'
            f'<span style="display:inline-block;width:16px;height:16px;background:{fill};'
            f'border:1px solid {border};margin-right:4px;flex-shrink:0;"></span>'
            f'<span style="font-size:11px;color:{text_color}">{label}</span>'
            f"</span>"
        )

    swatches_html = "".join(
        _swatch_html(label, fill, tc) for label, fill, tc in legend_items
    )
    if bsite_spans:
        swatches_html += _swatch_html(
            "Binding site region",
            fill="rgba(70,130,180,0.25)",
            border="steelblue",
        )

    legend_div = Div(
        text=(
            f'<div style="padding:6px 0 2px 0;font-weight:bold;font-size:12px;">'
            f"{legend_title}</div>"
            f'<div style="display:flex;flex-wrap:wrap;">{swatches_html}</div>'
        ),
        width=plot_width,
    )

    p = column(p1, p_blank, p2, legend_div)

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_name}.html"
    output_file(filename=str(out_path), title="Alignment result")
    save(p)

    return p, str(out_path)
