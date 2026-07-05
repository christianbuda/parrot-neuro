#!/usr/bin/env python3
"""Geometry-only artifact leadfields (eyes + muscle) — one shared DUNEuro transfer matrix.

The expensive part of a DUNEuro EEG solve, `computeEEGTransferMatrix`, depends only on the mesh,
conductivities, and electrodes — NOT on the dipoles. The artifact source groups (eyes, muscle) all
live on the same CGAL mesh and montage, so their transfer matrix is identical. This script computes
it **once** and then loops the cheap `applyEEGTransfer` per group, each group keeping its own
`valid_tissues` (eye compartment for eyes; Muscle+Skin for muscle) so dipoles snap into the correct
extra-brain shell instead of being dragged into grey matter.

Each group's output is the raw 3-component, average-referenced *geometric* leadfield (no brain
neural weighting — amplitude is applied downstream by the noise generator), rows = subject montage
(same electrode order as the brain leadfield), so the products are directly stackable:
    leadfields/sub-<S>/processed_duneuro<out_tag>-leadfield.npy   shape (n_elec, 3 * n_src)

Groups are described by --groups_json, a JSON list of
    {"name": ..., "dipoles_dir": <rel to output_dir>, "valid_tissues": [...], "out_tag": ...}
so the orchestrator can pass the mesh-appropriate tissue names (ITIS vs Sim4Life) without shell
quoting pain. Reuses the solver helpers from make_leadfield_duneuro.py (no duplicated physics).
"""
import argparse
import json
import os
import sys

import duneuropy as dp
import numpy as np
from scipy.spatial import cKDTree

from mesh_io import read_mesh, read_conductivities, read_tissues
from make_leadfield_duneuro import avg_ref, build_conductivity_config, read_electrodes, convert_dipoles

sys.stdout.reconfigure(line_buffering=True)


def snap_to_valid_tissue(positions_mm, nodes, tetrahedra, tissue_label, tissue_names, valid_names):
    """Place artifact dipoles at the nearest VALID-tissue tetrahedron centroid -> converted dipoles.

    Unlike the brain solver's get_dipoles (which finds each dipole's *containing* tet and only then
    snaps strays), warped artifact dipoles are NOT guaranteed to lie inside the mesh: the muscle warp
    places them against the scalp/skull surfaces, so a few land just outside the meshed volume (e.g.
    neck sources beyond the FOV). get_dipole_tissues' containment search grows its neighbour count
    unboundedly for such points until it trips cKDTree's missing-neighbour sentinel -> IndexError.

    For a geometry-only artifact leadfield we only need each dipole in the correct compartment, so we
    snap every dipole to the nearest centroid among the VALID-tissue tets with a single bounded k=1
    query. Robust (no unbounded loop), handles out-of-mesh points, and for a fine mesh moves already
    well-placed dipoles by less than a tet edge (sub-mm here). Returns free-orientation duneuro
    dipoles (3 orthogonal unit dipoles per source)."""
    valid_ids = []
    for name in [n.lower() for n in valid_names]:
        try:
            valid_ids.append(tissue_names.index(name))
        except ValueError:
            raise ValueError(f'valid tissue {name!r} not in mesh tissue names {tissue_names}')
    valid_mask = np.isin(tissue_label, valid_ids)
    if not valid_mask.any():
        raise ValueError(f'mesh has no tetrahedra in valid tissues {valid_names}')
    valid_centroids = nodes[tetrahedra[valid_mask]].mean(axis=1)   # (M,3) metres

    dip = np.load(positions_mm) / 1000.0                           # mm -> m
    dist, idx = cKDTree(valid_centroids).query(dip, k=1)
    snapped = valid_centroids[idx]
    moved_mm = dist * 1000.0
    print(f"  snapped {len(dip)} dipoles to nearest {valid_names} tet: "
          f"median {np.median(moved_mm):.2f} mm, max {moved_mm.max():.2f} mm "
          f"({int((moved_mm > 2).sum())} moved >2 mm, e.g. out-of-mesh sources).")
    return convert_dipoles(snapped)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--mesh_path', required=True)
    ap.add_argument('--tissue_names', required=True)
    ap.add_argument('--conductivities_path', required=True)
    ap.add_argument('--groups_json', default=None,
                    help='JSON list of {name, dipoles_dir, valid_tissues, out_tag}')
    ap.add_argument('--groups_json_file', default=None,
                    help='path to a file holding the same JSON (avoids shell-quoting tissue names)')
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()

    subject = args.subject
    output_dir = args.output_dir
    if args.groups_json_file:
        with open(args.groups_json_file) as f:
            groups = json.load(f)
    elif args.groups_json:
        groups = json.loads(args.groups_json)
    else:
        ap.error('one of --groups_json or --groups_json_file is required')

    # Fail fast: validate the group spec and that every group's dipoles exist BEFORE the expensive
    # driver/transfer-matrix build (~minutes), so a bad spec doesn't waste the whole solve.
    if not isinstance(groups, list) or not groups:
        sys.exit(f'group spec must be a non-empty JSON list, got {type(groups).__name__}')
    for g in groups:
        missing = {'name', 'dipoles_dir', 'valid_tissues', 'out_tag'} - set(g)
        if missing:
            sys.exit(f'group {g!r} missing keys: {sorted(missing)}')
        dp_path = os.path.join(output_dir, g['dipoles_dir'], 'dipole_positions.npy')
        if not os.path.exists(dp_path):
            sys.exit(f"group '{g['name']}' dipoles not found: {dp_path}")

    nodes, tetrahedra, tissue_label = read_mesh(args.mesh_path)
    tissue_names = read_tissues(args.tissue_names)
    conductivities = read_conductivities(args.conductivities_path)
    electrodes = read_electrodes(os.path.join(output_dir, f'electrodes/sub-{subject}/landmarks_10-5-full.csv'))

    # Snap + convert every group's dipoles onto the mesh BEFORE constructing the driver (the driver
    # reorders the grid arrays it is handed, so any mesh lookup must happen first).
    group_dipoles = {}
    for g in groups:
        print(f"\n=== Placing group '{g['name']}' (valid_tissues={g['valid_tissues']}) ===")
        dip_path = os.path.join(output_dir, g['dipoles_dir'], 'dipole_positions.npy')
        group_dipoles[g['name']] = snap_to_valid_tissue(dip_path, nodes, tetrahedra, tissue_label,
                                                        tissue_names, g['valid_tissues'])

    # Artifacts are always isotropic (no WM anisotropy relevant to eye/muscle sources).
    config = {
        'type': 'fitted', 'solver_type': 'cg', 'element_type': 'tetrahedron',
        'volume_conductor': {
            'grid': {'nodes': nodes, 'elements': tetrahedra},
            'tensors': build_conductivity_config(tissue_label, conductivities, dti_tensors_path=None),
        },
    }
    electrode_config = {'type': 'closest_subentity_center', 'codims': [3]}
    tm_config = {'solver': {'reduction': 1e-10}}
    if args.threads and args.threads > 0:
        tm_config['numberOfThreads'] = args.threads
    source_model_config = {
        'subtract_mean': True, 'post_process': False,
        'source_model': {
            'type': 'venant', 'restrict': True, 'numberOfMoments': 3, 'weightingExponent': 1,
            'relaxationFactor': 1e-6, 'initialization': 'closest_vertex',
            'referenceLength': 0.02, 'mixedMoments': True,
        },
    }

    print('Initializing DUNEuro driver...')
    driver = dp.MEEGDriver3d(config)
    print('Setting electrodes...')
    driver.setElectrodes(electrodes, electrode_config)
    print('Computing shared transfer matrix (once for all artifact groups)...')
    transfer_matrix, _ = driver.computeEEGTransferMatrix(tm_config)
    transfer_matrix = np.array(transfer_matrix)

    os.makedirs(os.path.join(output_dir, f'forwardsolvers/sub-{subject}'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'leadfields/sub-{subject}'), exist_ok=True)

    for g in groups:
        name, out_tag = g['name'], g['out_tag']
        print(f"\n=== Artifact group '{name}' leadfield ===")
        dipoles = group_dipoles[name]
        print('Applying transfer matrix...')
        leadfield, _ = driver.applyEEGTransfer(transfer_matrix, dipoles, source_model_config)
        leadfield = np.array(leadfield).T                          # (n_elec, 3 * n_src)

        np.save(os.path.join(output_dir, f'forwardsolvers/sub-{subject}/raw_duneuro{out_tag}-leadfield.npy'), leadfield)
        # Geometry-only: just average-reference (no volume/density/orientation-strength weighting).
        leadfield = avg_ref(leadfield)
        out = os.path.join(output_dir, f'leadfields/sub-{subject}/processed_duneuro{out_tag}-leadfield.npy')
        np.save(out, leadfield)
        print(f"  saved {leadfield.shape} -> {out}")


if __name__ == '__main__':
    main()
