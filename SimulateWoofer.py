from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import WooferXMLParser
import WooferRobot


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
for i in range(5000):
	# Update the woofer robot code
	tau = woofer.step(sim)
	if i%50 == 0:
		woofer.print_data()

	# Update the mujoco simulation
	sim.data.ctrl[:] = tau
	sim.step()
	viewer.render()


"""
Save data to file
"""
woofer.save_logs()