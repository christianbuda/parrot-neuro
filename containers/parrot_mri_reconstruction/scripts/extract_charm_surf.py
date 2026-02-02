import simnibs
from simnibs import mesh_io
import os
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Convert charm msh output to the boundary surfaces in .stl")
	parser.add_argument('--charm_dir', type=str, required = True, help='Path to the charm reconstruction folder')
	args = parser.parse_args()

	# Get the base directory from the command line
	charm_dir = args.charm_dir

	head_model = simnibs.read_msh(os.path.join(charm_dir, 'subject.msh'))

	# Use os.path.join to create the path to the bem directory.
	charm_dir = os.path.join(charm_dir, 'converted')
	os.mkdir(charm_dir)

	# extract interesting surfaces

	tissue_tag = 1001 # white matter
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'white.stl'))

	tissue_tag = 1002 # gray matter
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'gray.stl'))

	tissue_tag = 1003 # CSF
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'CSF.stl'))

	tissue_tag = 1005 # scalp
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'scalp.stl'))

	tissue_tag = 1006 # eyes
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'eyes_balls.stl'))

	tissue_tag = 1007 # compact bone
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'bone_compact.stl'))

	tissue_tag = 1008 # spongy bone
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'bone_spongy.stl'))

	tissue_tag = 1009 # blood
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'blood.stl'))

	tissue_tag = 1010 # eye muscles
	mesh_io.write_stl(head_model.crop_mesh(tags=[tissue_tag]), os.path.join(charm_dir, 'eyes_muscles.stl'))
