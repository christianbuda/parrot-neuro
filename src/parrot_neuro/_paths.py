"""Original on-disk paths, resolved through the manifest inventory.

File access requires registration and existence. Directory helpers only locate
containers; they do not establish output availability or scientific QC.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import _layout as L

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .subject import Subject


class SubjectPaths:
    def __init__(self, subject: "Subject"):
        self._s = subject

    # --- generic core -------------------------------------------------------
    def stage_dir(self, stage: str) -> Path:
        """``<derivatives>/<stage>/sub-<id>`` -- the per-subject folder of a stage."""
        return self._s.deriv / stage / self._s.subj

    def sfile(self, stage: str, *parts: str) -> Path:
        """A file inside a stage's per-subject folder."""
        return self._s.outputs.path(stage, name=Path(*parts).as_posix())

    # --- anatomy / volumes --------------------------------------------------
    def t1(self) -> Path:
        return self.sfile(L.RAW, "T1.nii.gz")

    def t2(self) -> Path:
        return self.sfile(L.RAW, "T2.nii.gz")

    def t1_stripped(self) -> Path:
        return self.sfile(L.SYNTHSTRIP, "T1_stripped.nii.gz")

    def t1_mask(self) -> Path:
        return self.sfile(L.SYNTHSTRIP, "T1_stripped_mask.nii.gz")

    def final_tissues(self) -> Path:
        return self.sfile(L.SIMNIBS, "final_tissues.nii.gz")

    def head_mesh(self) -> Path:
        """The SimNIBS charm volumetric head mesh (Gmsh ``.msh``)."""
        return self.sfile(L.SIMNIBS, "subject.msh")

    def tissue_labels(self, kind: str = "electrical", source: str = "simnibs") -> Path:
        """Tissue label volume. ``kind`` in {electrical, acoustic}; ``source`` in
        {simnibs, simnibs_itis, sim4life}."""
        return self.sfile(L.TISSUE, kind, f"{source}.nii.gz")

    def tissue_property(self, prop: str, kind: str = "electrical", source: str = "simnibs") -> Path:
        """A per-label property table (e.g. ``conductivities``, ``labels``, ``LUT``,
        or an acoustic property like ``density``)."""
        return self.sfile(L.TISSUE, kind, f"{source}_{prop}.txt")

    # --- recon backend dir (fastsurfer | freesurfer) ------------------------
    def recon_dir(self) -> Path:
        """The active surface-recon folder, resolved from ``surface_backend``."""
        if self._s.surface_backend is None:
            raise FileNotFoundError(f"No surface reconstruction for {self._s.subj}")
        return self.stage_dir(self._s.surface_backend)

    def color_lut(self) -> Path:
        self.recon_dir()
        return self.sfile(self._s.surface_backend, "FreeSurferColorLUT.txt")

    # --- atlas --------------------------------------------------------------
    def atlas(self, res: int) -> Path:
        return self.sfile(L.ATLAS, f"atlas{res}.nii.gz")

    def atlas_lut(self, res: int) -> Path:
        return self.sfile(L.ATLAS, f"atlas{res}_LUT.txt")

    def atlas_labels(self, res: int) -> Path:
        return self.sfile(L.ATLAS, f"atlas{res}_labels.txt")

    def atlas_aggregated(self) -> Path:
        return self.sfile(L.ATLAS, "atlas_aggregated.nii.gz")

    # --- surfaces (always 'freesurfer_'-prefixed, regardless of backend) ----
    def surface(self, name: str) -> Path:
        """A world-space surface by bare stem, e.g. ``charm_scalp`` or
        ``freesurfer_lh_middle`` -> ``surfaces/<subj>/<name>.ply``."""
        return self.sfile(L.SURFACES, f"{name}.ply")

    def cortex(self, hemi: str, layer: str = "middle") -> Path:
        """Cortical surface. ``hemi`` in {lh, rh}; ``layer`` in {pial, white, middle}."""
        return self.surface(f"freesurfer_{hemi}_{layer}")

    def bem(self, layer: str) -> Path:
        """BEM surface. ``layer`` in {brain, inner_skull, outer_skull, outer_skin}."""
        return self.surface(f"freesurfer_BEM_{layer}")

    def scalp(self) -> Path:
        return self.surface("charm_scalp")

    def vertex_attr(self, name: str) -> Path:
        """A per-vertex attribute array sitting next to a surface, e.g.
        ``freesurfer_lh_middle_thickness`` or ``freesurfer_lh_middle_original_labels_400``."""
        return self.sfile(L.SURFACES, f"{name}.npy")

    # --- forward model: dipoles / electrodes / mesh -------------------------
    def dipole_dir(self, spacing: float) -> Path:
        # Reconstruction parses spacing as a float before naming its folders.
        return self.stage_dir(L.DIPOLES) / f"spacing{float(spacing)}mm"

    def dipole_file(self, name: str, spacing: float) -> Path:
        """A dipole array, e.g. ``dipole_positions``, ``dipole_directions``,
        ``dipole_volume``, ``dipole_neural_density``, ``orient_type``,
        ``aggregated_dipole_labels``, or ``400Parcels_dipole_labels``."""
        return self.sfile(L.DIPOLES, f"spacing{float(spacing)}mm", f"{name}.npy")

    def dipole_block(self, spacing: float, block: str, name: str) -> Path:
        return self.sfile(L.DIPOLES, f"spacing{float(spacing)}mm", block, f"{name}.npy")

    def electrodes_csv(self) -> Path:
        return self.sfile(L.ELECTRODES, "landmarks_10-5-full.csv")

    def electrodes_selected(self) -> Path:
        return self.sfile(L.ELECTRODES, "selected_landmarks_10-5-full.json")

    def fiducials(self) -> Path:
        return self.sfile(L.SCALP, "fiducials.json")

    def tetmesh(self, ext: str = "mesh") -> Path:
        """The CGAL tetrahedral volume mesh. ``ext`` in {mesh, vtu}."""
        return self.sfile(L.TETMESH, f"tetrahedral_mesh.{ext}")

    # --- leadfields ---------------------------------------------------------
    def leadfield(self, key: str) -> Path:
        """A processed leadfield by key, e.g. ``duneuroCGAL-2.0mm``,
        ``openmeeg-4.0mm``, ``duneuroCGAL_anisotropic-2.0mm``,
        ``duneuro_artifact-eyes-CGAL``. Discover keys with
        :meth:`Subject.available_leadfields`."""
        return self._s.outputs.path("leadfield", key=key)

    # --- DWI tensor (optional) ----------------------------------------------
    def dwi_tensor(self, space: str = "T1") -> Path:
        return self.sfile(L.DWITENSOR, f"{self._s.subj}_space-{space}_model-dti_tensor.nii.gz")

    def dwi_param(self, param: str, space: str = "T1") -> Path:
        """A DTI-derived map. ``param`` in {fa, eigvals, eigvecs}."""
        return self.sfile(
            L.DWITENSOR, f"{self._s.subj}_space-{space}_model-dti_param-{param}.nii.gz"
        )

    def dwi_brain_mask(self) -> Path:
        return self.sfile(L.DWITENSOR, f"{self._s.subj}_space-T1_desc-brain_mask.nii.gz")

    def acpc_to_t1(self) -> Path:
        return self.sfile(L.DWITENSOR, f"{self._s.subj}_from-ACPC_to-T1_ras.txt")

    def tractogram(self, space: str = "T1") -> Path:
        return self.sfile(
            L.QSIRECON, "dwi", f"{self._s.subj}_space-{space}_model-ifod2_streamlines.tck.gz"
        )

    # --- connectivity -------------------------------------------------------
    def weights(self, n: int, normalized: bool = False) -> Path:
        """``normalized=True`` selects inverse-node-volume weights, not max scaling."""
        stem = "weights_invnodevol" if normalized else "weights"
        return self.sfile(L.CONNECTIVITY, f"{stem}_{n}.txt")

    def distances(self, n: int) -> Path:
        return self.sfile(L.CONNECTIVITY, f"distances_{n}.txt")

    def connectivity_atlas(self, n: int) -> Path:
        return self.sfile(L.CONNECTIVITY, f"atlas{n}_connectivity.nii.gz")

    def connectivity_labels(self, n: int) -> Path:
        """Connectome node names, one per line; line 0 == Unknown (dropped from the matrix)."""
        return self.sfile(L.CONNECTIVITY, f"labels_{n}.txt")

    def full_to_reduced(self, n: int) -> Path:
        """``reduced_id = full_to_reduced[full_atlas_id]`` (-1 if dropped from connectivity)."""
        return self.sfile(L.CONNECTIVITY, f"full_to_reduced_{n}.npy")

    def reduced_to_full(self, n: int) -> Path:
        """``full_atlas_id = reduced_to_full[reduced_id]`` (index 0 == Unknown)."""
        return self.sfile(L.CONNECTIVITY, f"reduced_to_full_{n}.npy")

    # --- anisotropy (optional) ----------------------------------------------
    def conductivity_tensors(self) -> Path:
        return self.sfile(L.ANISOTROPY, "conductivity_tensors.npy")

    def wm_element_indices(self) -> Path:
        return self.sfile(L.ANISOTROPY, "wm_element_indices.npy")

    # --- artifacts (optional; split registration/ + dipoles/ subtrees) ------
    def artifact_registration_dir(self) -> Path:
        return self._s.deriv / L.ARTIFACTS / "registration" / self._s.subj

    def artifact_dipoles_dir(self) -> Path:
        return self._s.deriv / L.ARTIFACTS / "dipoles" / self._s.subj

    def artifact_affine(self, direction: str = "mni_to_subject") -> Path:
        """``direction`` in {mni_to_subject, subject_to_mni}."""
        return self._s.outputs.path("artifacts", name=f"registration/{self._s.subj}/{direction}_affine.npy")

    def artifact_sources(self) -> Path:
        return self._s.outputs.path("artifacts", name=f"dipoles/{self._s.subj}/artifactsources.json")

    def artifact_dipole_file(self, name: str, kind: str = "eyes") -> Path:
        """``kind`` in {eyes, muscle}; ``name`` e.g. ``dipole_positions``,
        ``dipole_preferential_direction``, ``dipole_labels``, ``orient_type``."""
        return self._s.outputs.path("artifacts", name=f"dipoles/{self._s.subj}/{kind}/{name}.npy")

    # --- staged inputs: EEG / fMRI (optional) -------------------------------
    def eeg(self, task: str = "eyesclosed") -> Path:
        """``task`` in {eyesopen, eyesclosed}. Splice-free segments (.npz)."""
        return self.sfile(L.EEG, f"{self._s.subj}_task-{task}_eeg.npz")

    def eeg_metadata(self, task: str = "eyesclosed") -> Path:
        return self.sfile(L.EEG, f"{self._s.subj}_task-{task}_eeg.json")

    def fmri_timeseries(self, variant: str = "full", task: str = "rest") -> Path:
        """Schaefer atlas BOLD time series. ``variant`` in {full, conn}."""
        return self.sfile(
            L.FMRI, f"{self._s.subj}_task-{task}_atlas-schaefer_desc-{variant}_timeseries.npz"
        )

    def fmri_bold(self, task: str = "rest") -> Path:
        return self.sfile(
            L.FMRI, f"{self._s.subj}_task-{task}_space-native_desc-preproc_bold.nii.gz"
        )

    def optim_nodes(self, task: str = "rest") -> Path:
        """fMRI-derived optimization node mask over the connectome axis (.npz); keys
        ``keep_{N}``, ``optim_to_conn_{N}``, ``conn_to_optim_{N}``."""
        return self.sfile(
            L.FMRI, f"{self._s.subj}_task-{task}_atlas-schaefer_desc-optim_nodes.npz"
        )
