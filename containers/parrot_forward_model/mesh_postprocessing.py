import os
import nibabel as nib
import meshio
import numpy as np
import argparse

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
    
    return vertices, vertex_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels

def write_medit(output_path, vertices, vertex_labels, triangles, triangle_labels, tetrahedra, tetrahedron_labels):
    # reads medit file outputted by CGAL 6.1.1 mesher
    
    with open(output_path, 'w') as f:
        f.write('MeshVersionFormatted 1\n')
        f.write('Dimension 3\n')
        f.write('# CGAL::Mesh_complex_3_in_triangulation_3\n')

        triangles = np.concatenate([triangles+1, triangle_labels[:,np.newaxis]], axis = -1)
        tetrahedra = np.concatenate([tetrahedra+1, tetrahedron_labels[:,np.newaxis]], axis = -1)
        
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
    tetrahedra = tetrahedra[tetrahedron_labels!=0]
    tetrahedron_labels = tetrahedron_labels[tetrahedron_labels!=0]

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
