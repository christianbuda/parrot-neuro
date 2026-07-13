import os
import nibabel as nib
import meshio
import numpy as np
import argparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def read_medit(input_path):
    # reads medit file outputted by CGAL 6.1.1 mesher
    
    with open(input_path, 'r') as f:
        mesh = f.readlines()

    mesh = list(map(lambda x: x.strip(), mesh))

    assert mesh[0] == 'MeshVersionFormatted 1', 'Mesh file header not as expected, check pls.'
    assert mesh[1] == 'Dimension 3', 'Mesh file header not as expected, check pls.'
    assert mesh[2] == '# CGAL::Mesh_complex_3_in_triangulation_3', 'Mesh file header not as expected, check pls.'
    assert mesh[3] == 'Vertices', 'Mesh file header not as expected, check pls.'

    nvertices = int(mesh[4])
    vertices = mesh[5:5+nvertices]
    vertex_labels = np.array(list(map(lambda x: int(x.split(' ')[-1]), vertices)))
    vertices = np.array(list(map(lambda x: x.split(' ')[:-1], vertices))).astype(float)

    assert mesh[5+nvertices] == 'Triangles', 'Mesh file header not as expected, check pls.'
    ntriangles = int(mesh[6+nvertices])
    triangles = mesh[7+nvertices:7+nvertices+ntriangles]
    triangles = np.array(list(map(lambda x: x.split(' '), triangles))).astype(int)
    triangle_labels = triangles[:,-1]
    triangles = triangles[:,:-1]

    assert mesh[7+nvertices+ntriangles] == 'Tetrahedra', 'Mesh file header not as expected, check pls.'
    ntetrahedra = int(mesh[8+nvertices+ntriangles])
    tetrahedra = mesh[9+nvertices+ntriangles:9+nvertices+ntriangles+ntetrahedra]
    tetrahedra = np.array(list(map(lambda x: x.split(' '), tetrahedra))).astype(int)
    tetrahedron_labels = tetrahedra[:,-1]
    tetrahedra = tetrahedra[:,:-1]

    assert mesh[9+nvertices+ntriangles+ntetrahedra] == 'End', 'Mesh file header not as expected, check pls.'
    
    # medit files start counting from 1
    triangles -= 1
    tetrahedra -= 1
    
    return vertices, vertex_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels

def write_medit(output_path, vertices, vertex_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels):
    # reads medit file outputted by CGAL 6.1.1 mesher
    
    with open(output_path, 'w') as f:
        f.write('MeshVersionFormatted 1\n')
        f.write('Dimension 3\n')
        f.write('# CGAL::Mesh_complex_3_in_triangulation_3\n')

        # medit files start counting from 1
        triangles = triangles+1
        tetrahedra = tetrahedra+1

        triangles = np.concatenate([triangles, triangle_labels[:,np.newaxis]], axis = -1)
        tetrahedra = np.concatenate([tetrahedra, tetrahedron_labels[:,np.newaxis]], axis = -1)
        
        f.write('Vertices\n')
        f.write(str(len(vertices))+'\n')
        for idx, row in enumerate(vertices):
            f.write(' '.join(map(lambda x: f"{x:.17g}", row.tolist()))+f' {vertex_labels[idx]}\n')

        f.write('Triangles\n')
        f.write(str(len(triangles))+'\n')
        for idx, row in enumerate(triangles):
            f.write(' '.join(map(lambda x: str(x), row.tolist()))+'\n')

        f.write('Tetrahedra\n')
        f.write(str(len(tetrahedra))+'\n')
        for idx, row in enumerate(tetrahedra):
            f.write(' '.join(map(lambda x: str(x), row.tolist()))+'\n')

        f.write('End\n')
    
    return


def largest_connected_component(n_points, tetrahedra, tetrahedron_labels):
    """Keep only the tetrahedra of the mesh's largest connected component.

    CGAL meshing can pinch off tiny islands of tetrahedra (e.g. a few
    skull-cortical tets) disconnected from the main head volume. Each floating
    component is a null space in the FEM stiffness matrix, so DUNEuro's CG solve
    diverges to defect=nan or stagnates for hours. Returns the tets (+ labels) of
    the single largest component and diagnostics (n_dropped_tets, n_tet_components);
    the caller's vertex compaction then removes the islands' now-orphan vertices.

    Connectivity is over the node graph induced by tet edges (two nodes are
    adjacent iff they share a tetrahedron). Vertices used by no tetrahedron
    (label-0 orphans) form singleton components and are ignored here.
    """
    e = tetrahedra
    pairs = np.vstack([e[:, [0, 1]], e[:, [0, 2]], e[:, [0, 3]],
                       e[:, [1, 2]], e[:, [1, 3]], e[:, [2, 3]]])
    g = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                   shape=(n_points, n_points))
    _, comp = connected_components(g, directed=False)
    tet_comp = comp[tetrahedra[:, 0]]          # a tet's 4 nodes all share one component
    tet_components = np.unique(tet_comp)
    if len(tet_components) <= 1:
        return tetrahedra, tetrahedron_labels, 0, len(tet_components)
    largest = int(np.bincount(tet_comp).argmax())
    keep = tet_comp == largest
    return tetrahedra[keep], tetrahedron_labels[keep], int((~keep).sum()), len(tet_components)


if __name__ == "__main__":
    ################ input parsing ##############
    parser = argparse.ArgumentParser(
        description="Converts mesh file to world space according to nifti affine, filters away label 0 (which CGAL uses to fill the convex hull), and optionally converts the mesh to vtu.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--reference_nifti',
        type=str,
        required=True,
        help='Path to reference nifti file'
    )

    parser.add_argument(
        '--mesh',
        type=str,
        required=True,
        help='Path to input mesh'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output mesh'
    )
    
    parser.add_argument(
        '--export_vtu',
        action='store_true',
        required=False,
        help='Whether to export the mesh in vtu as well'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    nifti = args.reference_nifti
    input = args.mesh
    output = args.output
    export_vtu = args.export_vtu
    
    output = os.path.splitext(output)[0]

    print('Reading MEDIT mesh...', flush=True)
    points, points_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels = read_medit(input)

    # load original T1 to realign mesh to world space
    nifti = nib.load(nifti)

    affine = nifti.affine
    zooms = np.array(nifti.header.get_zooms())[:3]

    # scale back from mm to voxel size
    points /= zooms

    # Apply transform
    points = (affine @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T
    points = points[:,:3]

    # filter out background tetrahedra that complete the convex hull
    keep_tet = tetrahedron_labels != 0
    tetrahedra = tetrahedra[keep_tet]
    tetrahedron_labels = tetrahedron_labels[keep_tet]

    # Drop disconnected islands, keeping only the largest connected component (see
    # largest_connected_component). Done BEFORE the vertex compaction below so the
    # islands' orphan vertices are removed together with the label-0 orphans.
    tetrahedra, tetrahedron_labels, n_island_tet, n_tet_comp = \
        largest_connected_component(len(points), tetrahedra, tetrahedron_labels)
    if n_island_tet > 0:
        print(f'Kept the largest of {n_tet_comp} connected components: dropped '
              f'{n_island_tet} tetrahedra in {n_tet_comp - 1} disconnected island(s).', flush=True)

    # Compact the vertex array after dropping the label-0 hull tets. Any vertex
    # that belonged ONLY to a removed background tet is now an orphan: still in
    # the Vertices block but referenced by no tetrahedron. DUNEuro reads only
    # nodes + tetrahedra, so each orphan becomes a zero-connectivity DOF -> an
    # empty, zero-diagonal stiffness row -> ISTL "index N not in compressed
    # array" mid transfer-matrix solve. Keep only vertices referenced by a
    # surviving tet and remap the tet (and surface-triangle) indices to match.
    used = np.unique(tetrahedra)                       # sorted old vertex ids still in use
    remap = -np.ones(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used))
    n_orphan = len(points) - len(used)

    points = points[used]
    points_labels = points_labels[used]
    tetrahedra = remap[tetrahedra]

    # Surface triangles reference vertices too; a triangle touching a dropped
    # vertex belonged to the removed hull -> drop it, then remap the survivors.
    tri_keep = np.all(remap[triangles] >= 0, axis=1)
    n_tri_dropped = int((~tri_keep).sum())
    triangles = remap[triangles[tri_keep]]
    triangle_labels = triangle_labels[tri_keep]

    print(f'Compacted mesh: removed {n_orphan} orphan vertices and '
          f'{n_tri_dropped} dangling background triangles.', flush=True)

    print('Writing MEDIT mesh...', flush=True)
    write_medit(output+'.mesh', points, points_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels)

    if export_vtu:
        print('Exporting VTU mesh...', flush=True)
        cells = [
            ("triangle", triangles.tolist()),
            ("tetra", tetrahedra.tolist()),
        ]

        mesh = meshio.Mesh(
            points,
            cells,
            cell_data={"label": [triangle_labels, tetrahedron_labels]},
        )
        
        meshio.write(output+'.vtu', mesh)
