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

	(beta, theta, r) = (sim.data.qpos[7], sim.data.qpos[8], sim.data.qpos[9])

	r_fr_rel = np.zeros(3)
	r_fr_rel[0] = -(WOOFER_CONFIG.LEG_L + r)*np.sin(theta)
	r_fr_rel[1] = (WOOFER_CONFIG.LEG_L + r)*np.cos(theta)*np.sin(beta)
	r_fr_rel[2] = -(WOOFER_CONFIG.LEG_L + r)*np.cos(theta)*np.cos(beta)

	# r_fr_me = sim.data.qpos[0:3] + rotations.quat2mat(sim.data.qpos[3:7]) @ (r_fr_i + r_fr_rel)

	# using WooferDynamics LegJacobian2
	v_b_true = rotations.quat2mat(sim.data.qpos[3:7]) @ sim.data.qvel[0:3]
	v_i = v_b_true + MathUtils.CrossProductMatrix(sim.data.qvel[3:6]) @ (r_fr + r_fr_rel)


	v_rel = -WooferDynamics.LegJacobian(sim.data.qpos[7], sim.data.qpos[8], sim.data.qpos[9]) @ sim.data.qvel[6:9]

	e += (v_i - v_rel).T @ (v_i - v_rel)

	# Run the control law according to the control rate
	# if i%control_rate == 0:
	# 	# Add latency and noise
	# 	# Note: Noise is only re-sampled when a new control is applied
	# 	sim.data.ctrl[:] = tau_history[:,max(0,i-latency)] + np.random.randn(12) * tau_noise

	sim.data.ctrl[:] = tau

	sim.step()
	viewer.render()
print(e/timesteps)

"""
Save data to file
"""
woofer.save_logs()
print("Done Saving Logs")
