"""Prepare per-subject connectivity atlases from the full reconstruction atlas.

For each Schaefer resolution N in {100..1000}, this takes the full subject atlas
(``atlas/sub-<ID>/atlas{N}.nii.gz`` + ``atlas{N}_labels.txt``, both produced by
make_atlas.py in subject/T1w space) and collapses it into the connectivity
parcellation used downstream:

  * fine subdivisions (thalamic nuclei, amygdala subnuclei) are merged into the
    connectivity units expected by the group-average template, and
  * non-grey / non-cortical structures that should not carry connectivity
    (commissures, fornix, ventricles, optic structures, ...) are routed into
    ``Unknown`` (label 0).

The atlas is then renumbered to contiguous labels ``0..M`` (0 == Unknown), which
``tck2connectome`` consumes directly. Outputs, written to
``connectivity/sub-<ID>/`` and matching the template fallback byte-for-byte:

  * ``atlas{N}_connectivity.nii.gz`` -- renumbered atlas (tck2connectome input)
  * ``labels_{N}.txt``              -- one region name per line, index 0 == Unknown
  * ``reduced_to_full_{N}.npy``     -- full_id = reduced_to_full[reduced_id]
  * ``full_to_reduced_{N}.npy``     -- reduced_id = full_to_reduced[full_id] (-1 if dropped)

The connectivity matrices themselves (weights / distances) are M x M with the
Unknown / node-0 row dropped, as tck2connectome does by default -- so the matrix
row ``i`` corresponds to reduced label ``i + 1`` (i.e. ``labels_{N}.txt[i + 1]``).

Aggregation logic ported verbatim from development/connectivity_atlas.ipynb.
"""

import os
import argparse

import numpy as np
import nibabel as nib

RESOLUTIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_aggregation_dict():
    """Regions to merge/drop before building the connectivity matrix.

    Keys are the *target* connectivity units; each value lists the region names
    that should be folded into that target. The ``Unknown`` target collects
    everything that should be dropped from connectivity entirely.

    Ported verbatim from development/connectivity_atlas.ipynb -- this is a
    neuroscience grouping decision, not an implementation detail.
    """
    aggregate_regions = {
        'Unknown': [
            'Vermis-White-Matter', 'SCP', 'Fornix', 'CC_Posterior', 'CC_Mid_Posterior',
            'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior', 'Ant-Commisure', 'Third-Ventricle',
            'R-Fornix', 'L-Fornix', 'Left-cysts', 'Right-cysts', 'R-N.opticus', 'L-N.opticus',
            'R-Optic-tract', 'L-Optic-tract', 'R-Chiasma-Opticum', 'L-Chiasma-Opticum',
            'Epiphysis', 'Hypophysis', 'Infundibulum', 'DCG', 'Vermis', 'Floculus',
            'Left-Lateral-nucleus-olfactory-tract', 'Right-Lateral-nucleus-olfactory-tract',
            'Left-SRLM', 'Right-SRLM', 'Left-Fusion-amygdala-HP-FAH', 'Right-Fusion-amygdala-HP-FAH',
            'Left-Envelope-Amygdala', 'Right-Envelope-Amygdala', 'Left-Extranuclear-Amydala',
            'Right-Extranuclear-Amydala', 'Left-VentralDC', 'Right-VentralDC', 'Left-R', 'Right-R'
        ],
        'Pons': ['brainstem'],
        'Left-CeM': ['Left-CL', 'Left-CM', 'Left-L-Sg', 'Left-Pf', 'Left-Pc'],
        'Left-VA': ['Left-VAmc', 'Left-VM'],
        'Left-VLa': ['Left-VLp'],
        'Left-LD': ['Left-LP'],
        'Left-MDm': ['Left-Pt', 'Left-PaV'],
        'Left-PuM': ['Left-PuMm', 'Left-PuMl'],
        'Right-CeM': ['Right-CL', 'Right-CM', 'Right-L-Sg', 'Right-Pf', 'Right-Pc'],
        'Right-VA': ['Right-VAmc', 'Right-VM'],
        'Right-VLa': ['Right-VLp'],
        'Right-LD': ['Right-LP'],
        'Right-MDm': ['Right-Pt', 'Right-PaV'],
        'Right-PuM': ['Right-PuMm', 'Right-PuMl'],
        'Left-Basal-nucleus': ['Left-Basolateral-nucleus'],
        'Left-Central-nucleus': ['Left-Centromedial-nucleus'],
        'Left-Cortical-nucleus': [
            'Left-Fusion-amygdala-HP-FAH', 'Left-Hippocampal-amygdala-transition-HATA',
            'Left-Endopiriform-nucleus', 'Left-Lateral-nucleus-olfactory-tract',
            'Left-Intercalated-nucleus', 'Left-Prepiriform-cortex',
            'Left-Periamygdaloid-cortex', 'Left-Envelope-Amygdala', 'Left-Extranuclear-Amydala'
        ],
        'Right-Basal-nucleus': ['Right-Basolateral-nucleus'],
        'Right-Central-nucleus': ['Right-Centromedial-nucleus'],
        'Right-Cortical-nucleus': [
            'Right-Fusion-amygdala-HP-FAH', 'Right-Hippocampal-amygdala-transition-HATA',
            'Right-Endopiriform-nucleus', 'Right-Lateral-nucleus-olfactory-tract',
            'Right-Intercalated-nucleus', 'Right-Prepiriform-cortex',
            'Right-Periamygdaloid-cortex', 'Right-Envelope-Amygdala', 'Right-Extranuclear-Amydala'
        ],
    }
    return aggregate_regions


def load_labels(labels_path):
    """Parse an ``id,name`` label file (no header) into an {id: name} dict.

    Split on the first comma only, so region names are taken verbatim.
    """
    labels = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, name = line.split(',', 1)
            labels[int(key)] = name.strip()
    return labels


def build_connectivity_atlas(atlas, labels, aggregate_regions):
    """Collapse a full atlas volume into the connectivity parcellation.

    Parameters
    ----------
    atlas : np.ndarray (int)        full atlas label volume
    labels : dict {id: name}        names for every label declared in the atlas
    aggregate_regions : dict        target_name -> [source_name, ...]

    Returns
    -------
    reduced_atlas : np.ndarray (int32)   renumbered atlas, contiguous 0..M (0 == Unknown)
    reduced_to_full : np.ndarray         full_id = reduced_to_full[reduced_id]
    full_to_reduced : np.ndarray         reduced_id = full_to_reduced[full_id] (-1 if dropped)
    labels_list : list[str]              region name per reduced label (index 0 == Unknown)
    """
    present = set(np.unique(atlas).tolist())

    # Guard 1 (from the notebook): any label declared in the labels file but
    # absent from the volume must be one we intend to aggregate/drop -- otherwise
    # the atlas and the aggregation dict have drifted apart.
    missing_labels = {labels[v] for v in set(labels.keys()) - present}
    all_aggregated = set()
    for vals in aggregate_regions.values():
        all_aggregated.update(vals)
    not_handled = missing_labels - all_aggregated
    if not_handled:
        raise RuntimeError(
            "Atlas labels missing from the volume are not slated for aggregation; "
            f"aggregate them before building connectivity: {sorted(not_handled)}"
        )

    # Guard 2 (new): every name referenced by the aggregation dict (targets and
    # sources) must exist in the labels file, or the name->id lookup below would
    # KeyError cryptically. Fail loudly with the offending names instead.
    referenced = set(aggregate_regions.keys())
    for vals in aggregate_regions.values():
        referenced.update(vals)
    known_names = set(labels.values())
    unknown_names = referenced - known_names
    if unknown_names:
        raise RuntimeError(
            f"Aggregation names not found in the atlas labels: {sorted(unknown_names)}"
        )

    inverted_labels = {name: key for key, name in labels.items()}

    # Resolve names -> ids: {target_id: [source_id, ...]}
    aggregate_values = {}
    for target, sources in aggregate_regions.items():
        source_ids = [inverted_labels[s] for s in sources]
        aggregate_values[inverted_labels[target]] = source_ids

    atlas = atlas.copy()

    # Merge each source id into its target id in the volume. Sorted for
    # deterministic order (no source is itself another target, so no chaining).
    for key in sorted(aggregate_values.keys()):
        atlas[np.isin(atlas, aggregate_values[key])] = key

    # Renumber the remaining labels to contiguous 0..M (np.unique is sorted, so
    # Unknown==0 maps to reduced 0).
    old_labels = np.unique(atlas)
    new_labels = np.arange(len(old_labels))

    full_to_reduced = -np.ones(max(labels.keys()) + 1, dtype=int)
    full_to_reduced[old_labels] = new_labels
    # Also map the (now-absent) merged source ids to their target's reduced id,
    # so full_to_reduced is valid for every original label, not just survivors.
    for key in sorted(aggregate_values.keys()):
        full_to_reduced[aggregate_values[key]] = full_to_reduced[key]

    reduced_atlas = full_to_reduced[atlas].astype(np.int32)
    reduced_to_full = old_labels
    labels_list = [labels[int(x)] for x in old_labels]

    return reduced_atlas, reduced_to_full, full_to_reduced, labels_list


def main():
    parser = argparse.ArgumentParser(
        description="Build per-subject connectivity atlases (one per Schaefer resolution).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject identifier (e.g. "01")')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Derivatives root (e.g. /derivatives)')
    args = parser.parse_args()

    subject = args.subject
    output_dir = args.output_dir

    atlas_dir = os.path.join(output_dir, f'atlas/sub-{subject}')
    out_dir = os.path.join(output_dir, f'connectivity/sub-{subject}')
    os.makedirs(out_dir, exist_ok=True)

    aggregate_regions = get_aggregation_dict()

    for n in RESOLUTIONS:
        print(f'Building connectivity atlas for resolution {n}...', flush=True)

        atlas_img = nib.load(os.path.join(atlas_dir, f'atlas{n}.nii.gz'))
        atlas = atlas_img.get_fdata().astype(int)
        labels = load_labels(os.path.join(atlas_dir, f'atlas{n}_labels.txt'))

        reduced_atlas, reduced_to_full, full_to_reduced, labels_list = \
            build_connectivity_atlas(atlas, labels, aggregate_regions)

        out_img = nib.Nifti1Image(reduced_atlas, atlas_img.affine, atlas_img.header)
        out_img.set_data_dtype(np.int32)
        nib.save(out_img, os.path.join(out_dir, f'atlas{n}_connectivity.nii.gz'))

        np.save(os.path.join(out_dir, f'reduced_to_full_{n}.npy'), reduced_to_full)
        np.save(os.path.join(out_dir, f'full_to_reduced_{n}.npy'), full_to_reduced)

        # No trailing newline, to match the template fallback exactly.
        with open(os.path.join(out_dir, f'labels_{n}.txt'), 'w') as f:
            f.write('\n'.join(labels_list))

        print(f'  -> {len(labels_list)} regions (incl. Unknown).', flush=True)


if __name__ == "__main__":
    main()
