from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
np.set_printoptions(suppress=True)

import woofer_xml_parser

model = load_model_from_path("woofer_out.xml")
print(model)
sim = MjSim(model)
viewer = MjViewer(sim)

sim.step()

print("Nicely exposed function:\n")
print(sim.model.get_xml())
# viewer.render()





