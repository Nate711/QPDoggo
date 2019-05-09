import os
import shutil
from os.path import expanduser
from WooferConfig import WOOFER_CONFIG, ENVIRONMENT_CONFIG

def Parse():
	###### ROBOT PARAMETERS #####

	## Solver params ##
	woofer_timestep 		= ENVIRONMENT_CONFIG.DT	# timestep
	woofer_joint_solref 	= "0.001 1"			# time constant and damping ratio for joints
	woofer_joint_solimp 	= "0.9 0.95 0.001"	# joint constraint parameters

	woofer_geom_solref 		= "0.005 2"			# time constant and damping ratio for geom contacts
	woofer_geom_solimp 		= "0.9 0.95 0.001"	# geometry contact parameters

	woofer_armature 		= 0.0024			# armature for joints [kgm2]

	## Geometry params ##
	woofer_leg_radius = WOOFER_CONFIG.FOOT_RADIUS # radius of leg capsule
	woofer_friction = ENVIRONMENT_CONFIG.MU	# friction between legs and ground
	woofer_half_size = "%s %s %s"%(WOOFER_CONFIG.L/2, WOOFER_CONFIG.W/2, WOOFER_CONFIG.T/2) # half-size of body box

	woofer_leg_geom = "0 0 0 0 0 %s"%(-WOOFER_CONFIG.LEG_L) # to-from leg geometry

	woofer_start_position = "0 0 %s"%(WOOFER_CONFIG.LEG_L + woofer_leg_radius)	# Initial position of the robot torso

	## Joint params ##
	woofer_joint_range = "%s %s"		%(-WOOFER_CONFIG.REVOLUTE_RANGE, 	WOOFER_CONFIG.REVOLUTE_RANGE)	# joint range in rads for angular joints
	woofer_joint_force_range = "%s %s"	%(-WOOFER_CONFIG.MAX_JOINT_TORQUE, 	WOOFER_CONFIG.MAX_JOINT_TORQUE) # force range for ab/ad and forward/back angular joints
	woofer_ext_force_range = "%s %s"	%(-WOOFER_CONFIG.MAX_LEG_FORCE, 	WOOFER_CONFIG.MAX_LEG_FORCE) 	# force range for radial/extension joint
	woofer_ext_range = "%s %s"			%(-WOOFER_CONFIG.PRISMATIC_RANGE, 	WOOFER_CONFIG.PRISMATIC_RANGE) 	# joint range for radial/extension joint
	woofer_rad_damping = 15 	# damping on radial/extension joint [N/m/s]
	woofer_joint_damping = 0.2	# damping on ab/ad and f/b angular joints [Nm/rad/s]


	###### FILE PATHS  #####

	dir_path 	= os.path.dirname(os.path.realpath(__file__))
	in_file 	= dir_path+"/woofer.xml"
	out_file 	= dir_path + "/woofer_out.xml"

	### Parse the xml ###

	print('Parsing MuJoCo XML file:')
	print('Input xml: %s'%in_file)
	print('Output xml: %s'%out_file)

	with open(in_file, 'r') as file :
	  filedata = file.read()


	#### Replace variable names with values ####

	# Solver specs
	filedata = filedata.replace('woofer_timestep', str(woofer_timestep))
	filedata = filedata.replace('woofer_joint_solref', str(woofer_joint_solref))
	filedata = filedata.replace('woofer_geom_solref', str(woofer_geom_solref))
	filedata = filedata.replace('woofer_friction', str(woofer_friction))
	filedata = filedata.replace('woofer_armature', str(woofer_armature))
	filedata = filedata.replace('woofer_joint_solimp', str(woofer_joint_solimp))
	filedata = filedata.replace('woofer_geom_solimp', str(woofer_geom_solimp))

	# Joint specs
	filedata = filedata.replace('woofer_ext_force_range', str(woofer_ext_force_range))
	filedata = filedata.replace('woofer_ext_range', str(woofer_ext_range))
	filedata = filedata.replace('woofer_joint_range', str(woofer_joint_range))
	filedata = filedata.replace('woofer_joint_force_range', str(woofer_joint_force_range))
	filedata = filedata.replace('woofer_rad_damping', str(woofer_rad_damping))
	filedata = filedata.replace('woofer_joint_damping', str(woofer_joint_damping))

	# Geometry specs
	filedata = filedata.replace('woofer_mass', str(WOOFER_CONFIG.MASS))
	filedata = filedata.replace('woofer_leg_radius', str(woofer_leg_radius))
	filedata = filedata.replace('woofer_half_size', str(woofer_half_size))
	filedata = filedata.replace('woofer_leg_fb', str(WOOFER_CONFIG.LEG_FB))
	filedata = filedata.replace('woofer_leg_lr', str(WOOFER_CONFIG.LEG_LR))
	filedata = filedata.replace("woofer_leg_geom", str(woofer_leg_geom))
	filedata = filedata.replace("woofer_start_position", str(woofer_start_position))

	# Write the xml file
	with open(out_file, 'w') as file:
	  file.write(filedata)
