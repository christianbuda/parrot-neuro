"""Shared readers for the CGAL/SimNIBS tetrahedral mesh and its label tables.

Extracted from make_leadfield_duneuro.py so the leadfield solver and the
anisotropy front-end (dti_to_conductivity_tensors.py) read the mesh exactly the
same way, with no duneuro dependency. Keep this module dependency-light
(numpy + meshio only).
"""
import numpy as np
import meshio


def read_mesh(filename):
    """Return (nodes, tetrahedra, labels). Nodes are converted mm -> meters."""
    mesh = meshio.read(filename)

    # convert units to meters
    points = np.ascontiguousarray(mesh.points.astype(np.dtype('float64')) / 1000)
    tetrahedra = np.ascontiguousarray(mesh.cells_dict['tetra'].astype(np.dtype('int64')))
    if filename[-4:] == '.msh':
        labels = np.ascontiguousarray(mesh.cell_data['gmsh:physical'][1].astype(np.dtype('int64')))
    elif filename[-5:] == '.mesh':
        labels = np.ascontiguousarray(mesh.cell_data['medit:ref'][1].astype(np.dtype('int64')))
    else:
        raise ValueError('Wrong mesh input type (this is not a general reader!).')

    assert len(tetrahedra) == len(labels), 'Labels don\'t match tetrahedra, check reader!'
    return points, tetrahedra, labels


def read_conductivities(filename):
    """Read a `<label>,<value>` table; returns one float per label (S/m)."""
    with open(filename, 'r') as f:
        cond = f.readlines()

    cond = np.array(list(map(lambda x: float(x.split(',')[-1]), cond)))

    # Replace any absolute 0.0 conductivity with a tiny, safe number
    cond[np.isclose(cond, 0)] = 1e-6
    return cond


def read_tissues(filename):
    """Read a `<label>,<name>` table; returns lowercased names indexed by label."""
    with open(filename, 'r') as f:
        names = f.readlines()
    names = list(map(lambda x: x.split(',')[-1].strip().lower(), names))
    return names
