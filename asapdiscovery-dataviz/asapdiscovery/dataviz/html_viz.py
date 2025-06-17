import base64
import logging  # noqa: F401
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
from warnings import warn

import logomaker
import matplotlib.pyplot as plt
import pandas as pd
from airium import Airium
from asapdiscovery.data.backend.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    oechem,
    oemol_to_pdb_string,
    oemol_to_sdf_string,
    openeye_perceive_residues,
)
from asapdiscovery.dataviz.plip import (
    get_interactions_plip,
    make_color_res_subpockets,
)
from asapdiscovery.data.metadata.resources import active_site_chains, master_structures
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetTags,
    TargetVirusMap,
)
from asapdiscovery.data.util.dask_utils import backend_wrapper, dask_vmap
from asapdiscovery.data.util.logging import HiddenPrint
from asapdiscovery.dataviz._html_blocks import HTMLBlockData
from asapdiscovery.dataviz.visualizer import VisualizerBase
from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.modeling.modeling import superpose_molecule  # TODO: move to backend
from multimethod import multimethod
from pydantic.v1 import Field, root_validator

logger = logging.getLogger(__name__)


class ColorMethod(str, Enum):
    subpockets = "subpockets"


class HTMLVisualizer(VisualizerBase):
    """
    Class for generating HTML visualizations of poses.

    The main method is `visualize`, which takes a list of inputs and returns a list of HTML strings or writes them to disk, optionally a list of output paths can
    be provided.

    The `visualize` is heavily overloaded and can take a list of `DockingResult`, `Path`, `Complex`, or a tuple of `Complex` and a list of `Ligand`.

    Parameters
    ----------
    target : TargetTags
        Target to visualize poses for
    color_method : ColorMethod
        Protein surface coloring method. Can `subpockets` 
    debug : bool
        Whether to run in debug mode
    write_to_disk : bool
        Whether to write the HTML files to disk or return them as strings
    output_dir : Path
        Output directory to write HTML files to
    align : bool
        Whether to align the poses to the reference protein

    """

    target: TargetTags = Field(..., description="Target to visualize poses for")
    color_method: ColorMethod = Field(
        ColorMethod.subpockets,
        description="Protein surface coloring method. Can be `subpockets`",
    )
    debug: bool = Field(False, description="Whether to run in debug mode")
    write_to_disk: bool = Field(
        True, description="Whether to write the HTML files to disk"
    )
    output_dir: Path = Field(
        "html", description="Output directory to write HTML files to"
    )
    align: bool = Field(
        True, description="Whether to align the poses to the reference protein"
    )
    ref_chain: Optional[str] = Field(
        None, description="Reference chain ID to align to."
    )
    active_site_chain: Optional[str] = Field(
        None, description="Mobile chain ID to align."
    )
    reference_protein: Optional[Any]

    @root_validator(pre=True)
    def check_and_set_chains(cls, values):
        active_site_chain = values.get("active_site_chain")
        ref_chain = values.get("ref_chain")
        target = values.get("target")
        if not active_site_chain:
            values["active_site_chain"] = active_site_chains[target]
        # set same chain for active site if not specified
        if not ref_chain:
            values["ref_chain"] = active_site_chains[target]
        return values

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_protein = load_openeye_pdb(master_structures[self.target])

    @root_validator
    @classmethod

    def get_tag_for_color_method(self):
        """
        Get the tag to use for the color method.
        """
        if self.color_method == ColorMethod.subpockets:
            return DockingResultCols.HTML_PATH_POSE.value

    def get_color_dict(self, protein) -> dict:
        """
        Depending on color type, return a dict that contains residues per color.
        """
        if self.color_method == "subpockets":
            return make_color_res_subpockets(protein, self.target)
        else:
            raise ValueError(
                f"Unknown color method: {self.color_method}, must be 'subpockets'"
                )

    @dask_vmap(["inputs"], has_failure_mode=True)
    @backend_wrapper("inputs")
    def _visualize(
        self,
        inputs: list[DockingResult],
        outpaths: Optional[list[Path]] = None,
        **kwargs,
    ) -> list[dict[str, str]]:
        """
        Visualize a list of docking results. Calls the `_dispatch` method to handle the different input types.
        """
        if outpaths:
            if len(outpaths) != len(inputs):
                raise ValueError("outpaths must be the same length as inputs")

        if outpaths and not self.write_to_disk:
            warn("outpaths provided but write_to_disk is False. Ignoring outpaths.")
            outpaths = None

        return self._dispatch(inputs, outpaths=outpaths, **kwargs)

    def provenance(self):
        return {}

    @multimethod
    def _dispatch(
        self,
        inputs: list[DockingResult],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
        **kwargs,
    ) -> Union[list[dict[str, str]], list[str]]:
        """
        Implementation for a list of DockingResult objects.

        Parameters
        ----------
        inputs : list[DockingResult]
            List of DockingResult objects
        outpaths : Optional[list[Path]], optional
            List of output paths, by default None

        Returns
        -------
        Union[list[dict[str, str]], list[str]]
            List of metadata dictionaries or list of HTML strings if write_to_disk is False
        """
        data = []
        viz_data = []

        for i, result in enumerate(inputs):
            try:
                output_pref = result.unique_name
                if self.write_to_disk:
                    if not outpaths:
                        outpath = self.output_dir / output_pref / "pose.html"
                    else:
                        outpath = self.output_dir / Path(outpaths[i])
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                else:
                    outpath = None  # we don't need to write to disk

                viz = self.html_pose_viz(
                    poses=[result.posed_ligand.to_oemol()], protein=result.to_protein()
                )
                viz_data.append(viz)
                # write to disk
                if self.write_to_disk:
                    self.write_html(viz, outpath)

                # make dataframe with ligand name, target name, and path to HTML
                row = {}
                row[DockingResultCols.LIGAND_ID.value] = (
                    result.input_pair.ligand.compound_name
                )
                row[DockingResultCols.TARGET_ID.value] = (
                    result.input_pair.complex.target.target_name
                )
                row[DockingResultCols.SMILES.value] = result.posed_ligand.smiles
                row[self.get_tag_for_color_method()] = outpath
                data.append(row)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {result.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )

        if self.write_to_disk:
            return data
        else:
            return viz_data

    @_dispatch.register
    def _dispatch(
        self, inputs: list[Path], outpaths: Optional[list[Path]] = None, **kwargs
    ) -> Union[list[dict[str, str]], list[str]]:
        """
        Implementation for a list of Path objects. Assumes that the Path objects are PDB files.
        in turn calls the Complex version of the dispatch.

        Parameters
        ----------
        inputs : list[Path]
            List of Path objects
        outpaths : Optional[list[Path]], optional
            List of output paths, by default None

        Returns
        -------
        Union[list[dict[str, str]], list[str]]
            List of metadata dictionaries or list of HTML strings if write_to_disk is False
        """
        complexes = [
            Complex.from_pdb(
                p,
                target_kwargs={"target_name": f"{p.stem}_target"},
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
            )
            for i, p in enumerate(inputs)
        ]
        # dispatch to complex version
        return self._dispatch(complexes, outpaths=outpaths, **kwargs)

    @_dispatch.register
    def _dispatch(
        self,
        inputs: list[Complex],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
        **kwargs,
    ) -> Union[list[dict[str, str]], list[str]]:
        """
        Implementation for a list of Complex objects.

        Parameters
        ----------
        inputs : list[Complex]
            List of Complex objects
        outpaths : Optional[list[Path]], optional
            List of output paths, by default None

        Returns
        -------
        Union[list[dict[str, str]], list[str]]
            List of metadata dictionaries or list of HTML strings if write_to_disk is False
        """
        data = []
        viz_data = []
        for i, cmplx in enumerate(inputs):
            try:
                if self.write_to_disk:
                    if not outpaths:
                        output_pref = cmplx.unique_name
                        outpath = self.output_dir / output_pref / "pose.html"
                    else:
                        outpath = self.output_dir / Path(outpaths[i])

                    outpath.parent.mkdir(parents=True, exist_ok=True)
                else:
                    outpath = None

                # make html string
                viz = self.html_pose_viz(
                    poses=[cmplx.ligand.to_oemol()], protein=cmplx.target.to_oemol()
                )
                viz_data.append(viz)
                # write to disk
                if self.write_to_disk:
                    self.write_html(viz, outpath)

                # make dataframe with ligand name, target name, and path to HTML
                row = {}
                row[DockingResultCols.LIGAND_ID.value] = cmplx.ligand.compound_name
                row[DockingResultCols.TARGET_ID.value] = cmplx.target.target_name
                row[DockingResultCols.SMILES.value] = cmplx.ligand.smiles
                row[self.get_tag_for_color_method()] = outpath
                data.append(row)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {cmplx.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )

        if self.write_to_disk:
            # if we are writing to disk, return the metadata
            return data
        else:
            # if we are not writing to disk, return the HTML strings
            return viz_data

    @_dispatch.register
    def _dispatch(
        self,
        inputs: list[tuple[Complex, list[Ligand]]],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
        **kwargs,
    ) -> Union[list[dict[str, str]], list[str]]:
        """
        Implementation for a list of tuples of Complex and list of Ligand objects. Each ligand in the list is visualized, making this perfect for
        visualizing multiple poses of a single ligand, or multiple ligands in a single complex.

        Parameters
        ----------
        inputs : list[tuple[Complex, list[Ligand]]]
            List of tuples of Complex and list of Ligand objects
        outpaths : Optional[list[Path]], optional
            List of output paths, by default None

        Returns
        -------
        Union[list[dict[str, str]], list[str]]
            List of metadata dictionaries or list of HTML strings if write_to_disk is False
        """
        data = []
        viz_data = []
        for i, inp in enumerate(inputs):
            try:
                cmplx, liglist = inp
                if self.write_to_disk:
                    if not outpaths:
                        output_pref = cmplx.unique_name
                        outpath = self.output_dir / output_pref / "pose.html"
                    else:
                        outpath = self.output_dir / Path(outpaths[i])
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                else:
                    outpath = None

                # make html string
                viz = self.html_pose_viz(
                    poses=[lig.to_oemol() for lig in liglist],
                    protein=cmplx.target.to_oemol(),
                )
                viz_data.append(viz)
                # write to disk
                if self.write_to_disk:
                    self.write_html(viz, outpath)

                # make dataframe with ligand name, target name, and path to HTML
                row = {}
                row[DockingResultCols.LIGAND_ID.value] = cmplx.ligand.compound_name
                row[DockingResultCols.TARGET_ID.value] = cmplx.target.target_name
                if len(liglist) > 1:
                    row[DockingResultCols.SMILES.value] = "MANY"
                else:
                    row[DockingResultCols.SMILES.value] = liglist[0].smiles
                row[self.get_tag_for_color_method()] = outpath
                data.append(row)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {cmplx.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )

        if self.write_to_disk:
            # if we are writing to disk, return the metadata
            return data
        else:
            # if we are not writing to disk, return the HTML strings
            return viz_data

    def html_pose_viz(
        self, poses: list[oechem.OEMolBase], protein: oechem.OEMolBase, **kwargs
    ) -> str:
        """
        Generate HTML visualization of poses, called by the `_dispatch` method.

        Parameters
        ----------
        poses : list[oechem.OEMolBase]
            List of poses to visualize
        protein : oechem.OEMolBase
            Protein to visualize

        Returns
        -------
        str
            HTML string
        """
        protein = openeye_perceive_residues(protein, preserve_all=True)
        if self.target == "EV-A71-Capsid" or self.target == "EV-D68-Capsid":
            # because capsid has an encapsulated ligand, we need to Z-clip.
            slab = "viewer.setSlab(-11, 50)\n"
        else:
            slab = ""

        for pose in poses:
            oechem.OESuppressHydrogens(
                pose, True, True
            )  # retain polar hydrogens and hydrogens on chiral centers

        a = Airium()

        # first check if we need to align the protein and ligand. This already happens during docking, but not
        # during pose_to_viz.py.
        if self.align:
            # merge
            complex = protein
            for pose in poses:
                complex = combine_protein_ligand(complex, pose)

            # align complex to master structure
            complex_aligned, _ = superpose_molecule(
                self.reference_protein,
                complex,
                ref_chain=self.ref_chain,
                mobile_chain=self.active_site_chain,
            )

            # get pose and protein back
            opts = oechem.OESplitMolComplexOptions()
            opts.SetProteinFilter(
                oechem.OEOrRoleSet(
                    oechem.OEMolComplexFilterFactory(
                        oechem.OEMolComplexFilterCategory_Protein
                    ),
                    oechem.OEMolComplexFilterFactory(
                        oechem.OEMolComplexFilterCategory_Peptide
                    ),
                )
            )
            # pretend that the binding site is huge so we make sure that we include all ligands.
            opts.SetMaxBindingSiteDist(1000)

            pose = oechem.OEGraphMol()
            protein = oechem.OEGraphMol()
            oechem.OESplitMolComplex(
                pose,
                protein,
                oechem.OEGraphMol(),
                oechem.OEGraphMol(),
                complex_aligned,
                opts,
            )

        else:
            # just combine into a single molecule
            _pose = oechem.OEGraphMol()
            for pose in poses:
                oechem.OEAddMols(_pose, pose)
            pose = _pose

        oechem.OESuppressHydrogens(
            pose, True, True
        )  # retain polar hydrogens and hydrogens on chiral centers
        # now prep the coloring function.
        surface_coloring = self.get_color_dict(protein)

        residue_coloring_function_js = ""
        start = True
        for color, residues in surface_coloring.items():
            residues = [
                f"'{res}'" for res in residues
            ]  # need to wrap the string in quotes *within* the JS code
            if start:
                residue_coloring_function_js += (
                    "if (["
                    + ",".join(residues)
                    + "].includes(atom.resi+'_'+atom.chain)){ \n return '"
                    + color
                    + "' \n "
                )
                start = False
            else:
                residue_coloring_function_js += (
                    "} else if (["
                    + ",".join(residues)
                    + "].includes(atom.resi+'_'+atom.chain)){ \n return '"
                    + color
                    + "' \n "
                )

        # start writing the HTML doc.
        a("<!DOCTYPE HTML>")
        with a.html(lang="en"):
            with a.head():
                a.meta(charset="utf-8")
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN",
                    src="https://code.jquery.com/jquery-3.2.1.slim.min.js",
                )
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-vFJXuSJphROIrBnz7yo7oB41mKfc8JzQZiCq4NCceLEaO4IHwicKwpJf9c9IpFgh",
                    src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.3/umd/popper.min.js",
                )
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-alpBpkh1PFOepccYVYDB4do5UnbKysX5WZXm3XxPqe5iKTfUKjNkCk9SaVuEZflJ",
                    src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/js/bootstrap.min.js",
                )
                a.link(
                    crossorigin="anonymous",
                    href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css",
                    integrity="sha384-PsH8R72JQ3SOdhVi3uxftmaW6Vc51MKb0q5P2rRUpPvrszuE4W1povHYgTpBfshb",
                    rel="stylesheet",
                )
                a.script(src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js")
                a.script(src="https://d3js.org/d3.v5.min.js")
                with a.style():
                    a(
                        "/* Dropdown Button */\n      .dropbtn {\n        background-color: #04AA6D;\n        color: white;\n        padding: 16px;\n        font-size: 16px;\n        border: none;\n        border-radius: 5;\n      }\n\n      /* The container <div> - needed to position the dropdown content */\n      .dropdown {\n        position: absolute;\n        display: inline-block;\n        left: 1%;\n        top: 1%;\n      }\n      .dropdown_ctcs {\n        position: absolute;\n        top: 7%;\n        left: 1%;\n        display: inline-block;\n      }\n\n  .dropdown_lgplts {\n        position: absolute;\n        top: 13%;\n        left: 1%;\n        display: inline-block;\n      }\n\n    /* Dropdown Content (Hidden by Default) */\n      .dropdown-content {\n        display: none;\n        position: relative;\n        background-color: #f1f1f1;\n        min-width: 160px;\n        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);\n        z-index: 1;\n      }\n\n      /* Links inside the dropdown */\n      .dropdown-content a {\n        color: black;\n        padding: 12px 16px;\n        text-decoration: none;\n        display: block;\n        cursor: default;\n      }\n                                                              \n      /* Show the dropdown menu on hover */\n      .dropdown:hover .dropdown-content {display: block;}\n      .dropdown_ctcs:hover .dropdown-content {display: block;}\n   .dropdown_lgplts:hover .dropdown-content {display: block;}\n   \n      /* Change the background color of the dropdown button when the dropdown content is shown */\n      .dropdown:hover .dropbtn {background-color: #3e8e41;}\n    \n\n      .viewerbox {\n        position: absolute;\n        width: 200px;\n        height: 100px;\n        padding: 10px;\n      }\n\n      .logoplotbox_unfit {\n        position: absolute;\n        top: 35%;\n        right:1%;\n        border: 5px solid black;\n      }\n      .logoplotbox_fit {\n        position: absolute;\n        top: 35%;\n        left:1%;\n        border: 5px solid black;\n      }"
                    )
                a("<!-- wrap the main JS block to please the frontend gods -->")
                with a.div(klass="box"):
                    a.div(id="gldiv", style="width: 100vw; height: 100vh;")

                # dropdowns. 
                
                show_logoplot_insert = hide_logoplot_insert = ""
                # drop-down buttons for subpocket view:
                a("<!-- show the top dropdown (surfaces) -->")
                with a.div(klass="dropdown"):
                    a.button(klass="dropbtn", _t="Key (Surfaces)")
                    with a.div(
                        klass="dropdown-content", style="text-align: center"
                    ):
                        a.a(
                            href="#",
                            _t="Protein residue surfaces are colored by subpockets, see<br /> notion -> asapdiscovery -> Computational Chemistry Core -><br /> Computational Chemsitry Core Reference Documents -> Canonical-views-of-target-structures",
                        )
                    with a.div(klass="dropdown-content"):
                        a.a(
                            href="#",
                            _t="⚪ : Residue in chain with binding pocket, but not part of binding pocket",
                        )
                        a.a(
                            href="#",
                            _t="⚫ : Residue not in chain with binding pocket",
                        )

                a("<!-- show the bottom dropdown (contacts) -->")
                with a.div(klass="dropdown_ctcs"):
                    a.button(klass="dropbtn", _t="Key (Contacts)")
                    with a.div(
                        klass="dropdown-content", style="text-align: center"
                    ):
                        a.a(
                            href="#",
                            _t="Ligand-protein contacts are shown as dashed lines colored as:",
                        )
                    with a.div(klass="dropdown-content"):
                        a.a(href="#", _t="Gray : Hydrophobic interaction")
                        a.a(href="#", _t="Blue : Hydrogen bond")
                        a.a(href="#", _t="Lilac : Water bridge")
                        a.a(href="#", _t="Yellow : Salt bridge")
                        a.a(href="#", _t="Green : pi-stacking")
                        a.a(href="#", _t="Orange : pi-cation interaction")
                        a.a(href="#", _t="Light-green : Halogen bond")
                        a.a(href="#", _t="Purple : Metal complex")

            with a.script():
                # function to show/hide the logoplots
                a(
                    'function showLogoPlots(resi, chain) {\n        var x = document.getElementById("fitDIV_"+resi+"_"+chain);\n        var y = document.getElementById("unfitDIV_"+resi+"_"+chain);\n        x.style.display = "block";\n        y.style.display = "block";\n\n      }\n      function hideLogoPlots(resi, chain) {\n        var x = document.getElementById("fitDIV_"+resi+"_"+chain);\n        var y = document.getElementById("unfitDIV_"+resi+"_"+chain);\n        x.style.display = "none";\n        y.style.display = "none";\n      }'
                )

                # function to show 3DMol viewer
                a(
                    'var viewer=$3Dmol.createViewer($("#gldiv"));\n \
                    var prot_pdb = `    '
                    + oemol_to_pdb_string(protein)
                    + "\n \
                    \n \
                    `;\n \
                    var lig_sdf =`  "
                    + oemol_to_sdf_string(pose)
                    + '\n \
                    `;       \n \
                        //////////////// set up system\n \
                        viewer.addModel(prot_pdb, "pdb") \n \
                        // set protein sticks and surface\n \
                        viewer.setStyle({model: 0}, {stick: {colorscheme: "whiteCarbon", radius:0.15}});\n \
                        // define a coloring function based on our residue ranges. We can\'t call .addSurface separate times because the surfaces won\'t be merged nicely. \n \
                        var colorAsSnake = function(atom) { \
                        '
                    + residue_coloring_function_js
                    + ' \
                                        }}; \
                        viewer.addSurface("MS", {colorfunc: colorAsSnake, opacity: 0.9}) \n \
                    \n \
                        viewer.setStyle({bonds: 0}, {sphere:{radius:0.5}}); //water molecules\n \
                    \n \
                        viewer.addModel(lig_sdf, "sdf")   \n \
                        // set ligand sticks\n \
                        viewer.setStyle({model: -1}, {stick: {colorscheme: "pinkCarbon"}});\n \
                    \n \
                        ////////////////// enable show residue number on hover\n \
                        viewer.setHoverable({}, true,\n \
                        function (atom, viewer, event, container) {\n \
                            if (!atom.label) {\n \
                                if (atom.chain === undefined){ \
                                    display_str = \'LIGAND\'; \
                                    } else { \
                                    display_str = atom.chain + \': \' +  atom.resn + atom.resi;'
                    + show_logoplot_insert
                    + "} \
                                atom.label = viewer.addLabel(display_str, { position: atom, backgroundColor: 'mintcream', fontColor: 'black' }); \
                            }\n \
                        },\n \
                        function (atom) {\n \
                            if (atom.label) {\n \
                                viewer.removeLabel(atom.label);\n \
                                delete atom.label;\n"
                    + hide_logoplot_insert
                    + "}\n \
                        }\n \
                        );\n \
                        viewer.setHoverDuration(100); // makes resn popup instant on hover\n \
                    \n \
                        //////////////// add protein-ligand interactions\n \
                        var intn_dict = "
                    + str(
                        get_interactions_plip(
                            protein, pose, self.color_method, self.target
                        )
                    )
                    + '\n \
                        for (const [_, intn] of Object.entries(intn_dict)) {\n \
                            viewer.addCylinder({start:{x:parseFloat(intn["lig_at_x"]),y:parseFloat(intn["lig_at_y"]),z:parseFloat(intn["lig_at_z"])},\n \
                                                    end:{x:parseFloat(intn["prot_at_x"]),y:parseFloat(intn["prot_at_y"]),z:parseFloat(intn["prot_at_z"])},\n \
                                                    radius:0.1,\n \
                                                    dashed:true,\n \
                                                    fromCap:2,\n \
                                                    toCap:2,\n \
                                                    color:intn["color"]},\n \
                                                    );\n \
                        }\n \
                    \n \
                        ////////////////// set the view correctly\n \
                        viewer.setBackgroundColor(0xffffffff);\n \
                        viewer.setView(\n \
                        '
                    + HTMLBlockData.get_orient()
                    + " \
                        )\n\
                        viewer.setZoomLimits(1,250) // prevent infinite zooming\n"
                    + slab
                    + " viewer.render();"
                )

        return str(a)

    @staticmethod
    def write_html(html, path) -> None:
        """
        Write HTML to a file.

        Parameters
        ----------
        html : str
            HTML to write.
        path : Path
            Path to write HTML to.
        """
        with open(path, "w") as f:
            f.write(html)
