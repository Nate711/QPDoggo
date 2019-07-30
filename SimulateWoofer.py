from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import WooferXMLParser
import WooferRobot
import numpy as np
from WooferConfig import WOOFER_CONFIG, ENVIRONMENT_CONFIG
import rotations
import MathUtils
import WooferDynamics


"""
Initailize MuJoCo
"""
WooferXMLParser.Parse()
model 	= load_model_from_path("woofer_out.xml")
sim 	= MjSim(model)
viewer 	= MjViewer(sim)


"""
Create woofer controller
"""
woofer = WooferRobot.MakeWoofer()


"""
Run the simulation
"""
timesteps = ENVIRONMENT_CONFIG.SIM_STEPS

# Latency options
latency 		= WOOFER_CONFIG.LATENCY 		# ms of latency between torque computation and application at the joint
control_rate 	= WOOFER_CONFIG.UPDATE_PERIOD	# ms between updates (not in Hz)
tau_history 	= np.zeros((12,timesteps))
tau_noise 		= WOOFER_CONFIG.JOINT_NOISE 	# Nm

r_fr = np.array([WOOFER_CONFIG.LEG_FB, 	-WOOFER_CONFIG.LEG_LR, 0])

e = 0
for i in range(timesteps):
	# Step the woofer controller and estimators forward
	tau = woofer.step(sim)
	tau_history[:,i] = tau

	if i%50 == 0:
		# pass
		woofer.print_data()

	# Run the control law according to the control rate
	# if i%control_rate == 0:
	# 	# Add latency and noise
	# 	# Note: Noise is only re-sampled when a new control is applied
	# 	sim.data.ctrl[:] = tau_history[:,max(0,i-latency)] + np.random.randn(12) * tau_noise

	sim.data.ctrl[:] = tau

	sim.step()
	viewer.render()

"""
Save data to file
"""
woofer.save_logs()
print("Done Saving Logs")
