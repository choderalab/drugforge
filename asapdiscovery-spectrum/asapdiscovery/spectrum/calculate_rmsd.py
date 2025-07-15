from pathlib import Path

import numpy as np
import pymol2
import MDAnalysis as mda
from Bio import pairwise2
from asapdiscovery.data.backend.openeye import load_openeye_pdb, save_openeye_pdb
from asapdiscovery.modeling.modeling import superpose_molecule
from asapdiscovery.spectrum.blast import pdb_to_seq

from typing import Union
import logging


def rmsd_alignment(
    target_pdb: str,
    ref_pdb: str,
    final_pdb: str,
    target_chain="A",
    ref_chain="A",
) -> tuple[float, Path]:
    """Calculate RMSD of a molecule against a reference

    Parameters
    ----------
    target_pdb : str
        Path to PDB of protein to align.
    ref_pdb : str
        Path to PDB to align target against.
    final_pdb : str
        Path to save PDB of aligned target.
    target_chain : str, optional
        The chain of target which will be used for alignment, by default "A"
    ref_chain : str, optional
        The chain of reference which will be used for alignment, by default "A"

    Returns
    -------
    float, str
       RMSD after alignment, Path to saved PDB
    """
    protein = load_openeye_pdb(target_pdb)
    ref_protein = load_openeye_pdb(ref_pdb)

    aligned_protein, rmsd = superpose_molecule(
        ref_protein, protein, ref_chain=ref_chain, mobile_chain=target_chain
    )
    pdb_aligned = save_openeye_pdb(aligned_protein, final_pdb)

    return rmsd, pdb_aligned


def select_best_colabfold(
    results_dir: str,
    seq_name: str,
    pdb_ref: str,
    chain="A",
    final_pdb="aligned_protein.pdb",
    fold_model="alphafold2_ptm",
) -> tuple[float, str]:
    """Select the best seed output (repetition) from a ColabFold run based on its RMSD wrt the reference.

    Parameters
    ----------
    results_dir : str
        The directory containing the ColabFold results.
    seq_name : str
        The name we gave to the sequence in the csv file.
    pdb_ref : str
        The path to the PDB of te reference protein.
    chain : str, optional
        Chain of both reference and generated PDB that will be used, by default "A"
    final_pdb : str, optional
        Path to the PDB where aligned structure will be saved, by default "aligned_protein.pdb"
    fold_model : str, optional
        The model used for ColabFold, by default "alphafold2_ptm"

    Returns
    -------
    Tuple[float, str]
        RMSD after alignment, Path to saved PDB

    Raises
    ------
    FileNotFoundError
        The directory with ColabFold results does not exist
    """

    rmsds = []
    seeds = []
    file_seed = []

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"A folder with ColbFold results {results_dir} does not exist"
        )

    for file_path in results_dir.glob(f"{seq_name}*_{fold_model}_model_1_seed_*.pdb"):
        pdb_to_compare = file_path
        seed = str(pdb_to_compare).split("_")[-1].split(".")[0]
        rmsd, pdb = rmsd_alignment(pdb_to_compare, pdb_ref, final_pdb, chain, chain)
        rmsds.append(rmsd)
        seeds.append(seed)
        file_seed.append(file_path)
        logging.info(f"RMSD for seed {seed} is {rmsd} A")

    if len(rmsds) == 0:
        logging.warning(f"No ColabFold entry for {seq_name} and model {fold_model} found.")
        return 0, ""
    min_rmsd = np.argmin(rmsds)
    min_rmsd_file = file_seed[min_rmsd]
    logging.info(
        f"{seq_name} seed with least RMSD is {seeds[min_rmsd]} with RMSD {rmsds[min_rmsd]} A"
    )

    min_rmsd, final_pdb = rmsd_alignment(
        min_rmsd_file, pdb_ref, final_pdb, chain, chain
    )

    return min_rmsd, str(final_pdb)


def save_alignment_pymol(
    pdbs: list,
    labels: list,
    reference: str,
    session_save: str,
    align_chain=str,
    color_by_rmsd=False,
) -> None:
    """Imports the provided PDBs into a Pymol session and saves

    Parameters
    ----------
    pdbs : list
        List with paths to pdb file to include.
    labels : list
        List with labels that will be used in protein objects.
    reference : str
        Path to reference PDB.
    session_save : str
        File name for the saved PyMOL session.
    align_chain : str
        Chain of ref to align target with.
    color_by_rmsd : bool, optional
        Option to color aligned targets by RMSD with respect to reference.
    """

    def hide_chain(p, chain, obj):
        """Hide the other chain from visualization in obj"""
        dimer_chains = {"A", "B"}
        hide_chain = (dimer_chains - {chain}).pop()
        p.cmd.select("chainb", f"{obj} and chain {hide_chain.upper()}")
        p.cmd.remove("chainb")
        p.cmd.delete("chainb")

    p = pymol2.PyMOL()
    p.start()

    p.cmd.load(reference, object="ref_protein")
    p.cmd.color("gray", "ref_protein")
    # Optionaly remove other chain from the reference protein
    if align_chain == "both":
        align_sel = ""
    else:
        align_sel = f" and chain {align_chain}"
        hide_chain(p, align_chain, "ref_protein")

    p.cmd.select("chaina", f"ref_protein{align_sel}")
    p.cmd.color("gray", "ref_protein")

    for i, pdb in enumerate(pdbs):
        if len(str(pdb)) > 0:
            # In case the entry is empty (when no CF output was found)
            pname = labels[i]
            p.cmd.load(pdb, object=pname)
            # PDBs should be aligned but in case they are not
            p.cmd.select("chainp", pname + align_sel)
            # It's better to align wrt a single chain than the whole protein (at least one binding site to compare)
            p.cmd.align(f"{pname} and chain A", "ref_protein and chain A")
            if color_by_rmsd:
                colorbyrmsd(p, "chainp", "chaina", minimum=0, maximum=2)
                p.cmd.color("red", "ref_protein")
            if len(align_chain) == 1:
                hide_chain(p, align_chain, pname)
            p.cmd.delete("chainp")
    p.cmd.delete("chaina")

    # set visualization
    p.cmd.set("bg_rgb", "white")
    p.cmd.bg_color("white")
    p.cmd.hide("everything")
    p.cmd.show("cartoon")
    p.cmd.set("transparency", 0.8)
    p.cmd.set("transparency", 0.3, "ref_protein")

    # Color ligand and binding site
    p.cmd.select("ligand", "resn UNK or resn LIG")
    p.cmd.select(
        "binding_site", "name CA within 5 of resn UNK or name CA within 5 resn LIG"
    )
    p.cmd.show("sticks", "ligand")
    p.cmd.color("red", "ligand")

    p.cmd.save(session_save)
    return

def convert_chain_id(chain: str) -> str:
    """Convert a chain identifier between letter and number representations.
    The ASCII values for lowercase letters start from a=97, so we subtract/add 96 to convert.

    Parameters
    ----------
    chain : str
        1 or 3-letter chain identifier.

    Returns
    -------
    str
        Converted chain identifier.
    """
    if chain.isalpha():
        return ord(chain.lower()) - 96
    elif chain.isdigit():
        return chr(int(chain) + 96)
    else:
        logging.warning(f"Chain identifier {chain} is not valid.")
        return chain

def colorbyrmsd(
    p: pymol2.PyMOL,
    target_sel: str,
    ref_sel: str,
    quiet=True,
    minimum=None,
    maximum=None,
):
    """Color aligned proteins by RMSD with respect to the target.
    Based on script by original authors Shivender Shandilya and Jason Vertrees,
    rewrite by Thomas Holder. License: BSD-2-Clause.
    http://pymolwiki.org/index.php/ColorByRMSD

    Parameters
    ----------
    p : pymol2.PyMOL
        Pymol session
    target_sel : str
        Selection of aligned target
    ref_sel : str
        Selection of reference protein
    quiet : bool, optional
        Not print RMSD info, by default True
    minimum : Union[int,float], optional
        Set a fixed min RMSD for coloring, by default None
    maximum : Union[int,float], optional
        Set a fixed max RMSD for coloring, by default None
    """
    from chempy import cpv

    selboth, aln = "both", "aln"
    p.cmd.align(target_sel, ref_sel, cycles=0, transform=0, object=aln)
    p.cmd.select(selboth, f"{target_sel} or {ref_sel}")

    idx2coords = {}
    p.cmd.iterate_state(
        -1, selboth, "idx2coords[model,index] = (x,y,z)", space=locals()
    )

    if p.cmd.count_atoms("?" + aln, 1, 1) == 0:
        p.cmd.refresh()

    b_dict = {}
    for col in p.cmd.get_raw_alignment(aln):
        assert len(col) == 2
        b = cpv.distance(idx2coords[col[0]], idx2coords[col[1]])
        for idx in col:
            b_dict[idx] = b

    p.cmd.alter(selboth, "b = b_dict.get((model, index), -1)", space=locals())

    p.cmd.orient(selboth)
    p.cmd.show_as("cartoon", "byobj " + selboth)
    p.cmd.color("gray", selboth)
    p.cmd.spectrum("b", "red_blue", selboth + " and b > -0.5", minimum, maximum)

    # Make colorbar
    if minimum is not None and maximum is not None:
        p.cmd.ramp_new("colorbar", "none", [minimum, maximum], ["red", "blue"])

    if not quiet:
        logging.info("ColorByRMSD: Minimum Distance: %.2f" % (min(b_dict.values())))
        logging.info("ColorByRMSD: Maximum Distance: %.2f" % (max(b_dict.values())))
        logging.info(
            "ColorByRMSD: Average Distance: %.2f" % (sum(b_dict.values()) / len(b_dict))
        )

    p.cmd.delete(aln)
    p.cmd.delete(selboth)

    return

def get_residue_mapping(seq_ref: str, seq_mob:str) -> tuple[int, int]:
    """Aligns two sequences and returns the correct start and end residue indices,
    ignoring gaps

    Parameters
    ----------
    seq_ref : str
        string of reference sequence
    seq_mob : str
        string of mobile sequence

    Returns
    -------
    tuple[int, int]
        start and end residue indices (1-based indexing)
    """
    alignments = pairwise2.align.globalxx(seq_ref, seq_mob)
    aligned_ref, aligned_mob = alignments[0][:2]  # Extract first alignment

    ref_idx, mob_idx = [], []
    r, m = 0, 0  # Residue counters (no gaps)

    for a, b in zip(aligned_ref, aligned_mob):
        ref_idx.append(r if a != "-" else None)
        mob_idx.append(m if b != "-" else None)
        r += a != "-"
        m += b != "-"

    # Find first and last matched residues
    start_idx = next(
        i
        for i in range(len(ref_idx))
        if ref_idx[i] is not None and mob_idx[i] is not None
    )
    end_idx = next(
        i
        for i in range(len(ref_idx) - 1, -1, -1)
        if ref_idx[i] is not None and mob_idx[i] is not None
    )

    return start_idx + 1, end_idx + 1  # 1-based indexing

def find_bsite_resids(
    pdb:str,
    pdb_ref:str,
    aligned_temp:str,
    ligres:str = "UNK",
    chain_m:str ="A",
    chain_r:str ="A",
    bsite_dist:float = 4.5,
    res_threshold: int = 5,
):
    """Find binding site residues in a protein-ligand complex based on ligand proximity.

    Parameters
    ----------
    pdb : str
        Path to PDB of protein to align.
    pdb_ref : str
        Path to reference PDB to align target against.
    aligned_temp : str
        Path to save aligned PDB.
    ligres : str, optional
        Residue name of ligand, by default "UNK"
    chain_m : str, optional
        Chain ID of mobile, by default "A"
    chain_r : str, optional
        Chain ID of ref, by default "A"
    bsite_dist : float, optional
        Distance to ligand in angstroms defining binding site, by default 4.5
    res_threshold : int, optional
        Threshold of residue IDs that can be assigned to binding site. Avoids asigning terminal atoms. By default 5

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of binding site residue IDs in mobile and reference structures.

    Raises
    ------
    ValueError
        No binding site residues have an idx above res_threshold
    """
    from scipy.spatial.distance import cdist
    rmsd, pdb_aln = rmsd_alignment(pdb, pdb_ref, aligned_temp, chain_m, chain_r)
    u = mda.Universe(pdb_aln)
    u_ref = mda.Universe(pdb_ref)

    # Initial atom selections
    bs_atoms = u_ref.select_atoms(
        f"protein and chainid {chain_r} and around {bsite_dist} resname {ligres}"
    )
    lig_atoms = u_ref.select_atoms(f"chainid {chain_r} and resname {ligres}")
    ca_mob = u.select_atoms(f"protein and chainid {chain_m} and name CA").atoms

    # Handle incorrect chain selections
    if len(ca_mob) == 0:
        logging.warning("The mobile chain is incorrect, attempting to fix")
        chain_m = convert_chain_id(chain_m)
        ca_mob = u.select_atoms(f"protein and chainid {chain_m} and name CA").atoms

    if len(bs_atoms) == 0:
        logging.warning("The reference chain is incorrect, attempting to fix")
        chain_r = convert_chain_id(chain_r)
        bs_atoms = u_ref.select_atoms(
            f"protein and chainid {chain_r} and around {bsite_dist} resname {ligres}"
        )
        lig_atoms = u_ref.select_atoms(f"chainid {chain_r} and resname {ligres}")

    ref_pos = lig_atoms.positions

    bs_ref = np.unique(bs_atoms.resids)
    bs_ref = bs_ref[
        bs_ref >= res_threshold
    ]  # In case terminal residue is categorized as binding site
    n_res = len(bs_ref)

    if len(bs_ref) == 0:
        raise ValueError(f"No binding site residues have an idx above {res_threshold}")

    distances = cdist(ca_mob.positions, ref_pos, metric="euclidean")

    sorted_flat = np.argsort(distances.ravel())
    rows, _ = np.unravel_index(sorted_flat, distances.shape)

    res_seen = set()
    bs_mob = [
        ca_mob.resids[r]
        for r in rows
        if (ca_mob.resids[r] not in res_seen and not res_seen.add(ca_mob.resids[r]))
    ]

    return np.sort(bs_mob[:n_res]), bs_ref

def get_binding_site_rmsd(
    file_mob: Union[Path, str],
    file_ref: Union[Path, str],
    bsite_dist:float = 4.5,
    rmsd_mode:str = "CA",
    chain_mob:str = "A",
    chain_ref:str = "A",
    ligres:str = "LIG",
    lig_ref_pdb:str = None,
    chain_ref2:str = "A",
    aligned_temp:str = None,
) -> float:
    """Calculate RMSD for the Binding Site residues between file_mob and file_ref
    The binding site is defined as all residues protein within bsite_dist Angs of the ligand in the reference protein.
    If the ligand is missing in the reference, an auxiliary reference (`lig_ref_pdb`) may be provided to define the binding site.

    Parameters
    ----------
    file_mob : Union[Path, str]
        Path to target file
    file_ref : Union[Path, str]
        Path to reference file
    bsite_dist : float, optional
        Cutoff distance to ligand in Angs, by default 4.5
    rmsd_mode : str, optional
        Type of RMSD to compute: 'CA' for alpha carbons or 'heavy' for all heavy atoms, by default "CA".
    chain_mob : str, optional
        Chain ID of target to compare, by default "1"
    chain_ref : str, optional
        Chain ID of reference to compare, by default "1"
    ligres : str, optional
        Resname of ligand residue, by default "UNK"
    lig_ref_pdb : Path, optional
       Optional Reference PDB to define binding site, in case ref doesn't have a ligand, by default None
    chain_ref2 : str, optional
        Optional reference Chain ID, used when the main reference doesn't have a ligand, by default "A"
    aligned_temp : Path, optional
        Optional folder to store alignment PDB when calculating bsite with additional PDB ref, by default None

    Returns
    -------
    float
        RMSD of binding site, or -1 if RMSD could not be calculated.
    Raises
    ------
    ValueError
        No ligand found in reference, and not second reference provided.
    """
    file_mob = str(file_mob)
    file_ref = str(file_ref)

    u = mda.Universe(file_mob).select_atoms(
        f"protein and segid {chain_mob} and not resname ACE and not resname NME"
    )
    u_ref = mda.Universe(file_ref).select_atoms(
        f"protein and segid {chain_ref} and not resname ACE and not resname NME"
    )

    u_ref_l = mda.Universe(file_ref).select_atoms(
        f"(protein and segid {chain_ref} and not resname ACE and not resname NME) or resname {ligres}"
    )
    u_lig = u_ref_l.select_atoms(f"resname {ligres}")
    bs_atoms = u_ref_l.select_atoms(f"protein and around {bsite_dist} resname {ligres}")

    # Handle reference with no ligand
    if len(u_lig) == 0:
        if lig_ref_pdb and aligned_temp:
            bs_ids, __ = find_bsite_resids(
                file_mob,
                lig_ref_pdb,
                aligned_temp,
                ligres,
                chain_mob,
                chain_ref2,
                bsite_dist,
                res_threshold=5,
            )
            bs_atoms = u_ref_l.select_atoms(" or ".join([f"resid {r}" for r in bs_ids]))
        else:
            raise ValueError(
                f"No ligand found in ref with resname {ligres}. Provide a correct ligand name or a second reference PDB."
            )

    # Align sequences to ensure residue numbering is consistent
    seq_mob = pdb_to_seq(Path(file_mob), chain=str(chain_mob)).seq.replace("X", "")
    seq_ref = pdb_to_seq(Path(file_ref), chain=str(chain_ref)).seq.replace("X", "")

    start_resid, end_resid = get_residue_mapping(seq_ref, seq_mob)
    u_ref_filter = u_ref.select_atoms(f"resid {start_resid}:{end_resid}")
    u = u.select_atoms(f"resid {start_resid}:{end_resid}")

    # Select binding site residues
    binding_site = [
        r.resid - start_resid + 1 for r in bs_atoms.residues if "CA" in r.atoms.names
    ]
    binding_site_n = [
        str(r.resname) for r in bs_atoms.residues if "CA" in r.atoms.names
    ]

    binding_site_m = []
    binding_site_r = []
    for i, r in enumerate(binding_site):
        res = u.residues[r - 1]
        if binding_site_n[i] == res.resname:
            binding_site_m.append(res.resid)
            binding_site_r.append(r)
        else:
            logging.warning(f"Missmatch in residue with index {i} {binding_site_n[i]} != {res.resname}")

    sel_bs = " or ".join(f"resid {r}" for r in binding_site_r)
    sel_bs_m = " or ".join(f"resid {r}" for r in binding_site_m)

    if len(sel_bs_m) > 0:
        u_bs_m = u.select_atoms(sel_bs_m)
    else:
        alignment = pairwise2.align.globalms(seq_mob, seq_ref, 2, -1, -0.8, -0.5)[0]
        logging.warning(
            "The sequences may be different! \n", pairwise2.format_alignment(*alignment)
        )
        return -1

    u_bs_ref = u_ref_filter.select_atoms(sel_bs)

    rmsd_sel = "name CA" if rmsd_mode == "CA" else "not name H*"
    m_pos, ref_pos = u_bs_m.select_atoms(rmsd_sel), u_bs_ref.select_atoms(rmsd_sel)

    # Match common atoms per residue
    mpos_list, refpos_list = [], []
    for mob_res, ref_res in zip(m_pos.residues, ref_pos.residues):
        common_atoms = set(mob_res.atoms.names).intersection(ref_res.atoms.names)
        mob_common, ref_common = mob_res.atoms.select_atoms(
            f"name {' or name '.join(common_atoms)}"
        ), ref_res.atoms.select_atoms(f"name {' or name '.join(common_atoms)}")
        if len(mob_common) == len(ref_common):
            mpos_list.append(mob_common.positions)
            refpos_list.append(ref_common.positions)

    try:
        rmsd = np.sqrt(
            ((np.vstack(mpos_list) - np.vstack(refpos_list)) ** 2).sum(-1).mean()
        )
    except ValueError:
        logging.error(f"Error: Mismatched lengths ({len(m_pos)} vs {len(ref_pos)})")
        rmsd = -1

    return rmsd