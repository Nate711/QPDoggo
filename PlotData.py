import matplotlib.pyplot as plt
import numpy as np

### Init graphics ###

data = np.load('SimulationData.npz')
torque_history = data['th']
force_history = data['fh']
ref_wrench_history = data['rwh']
contacts_history = data['ch']
active_feet_history = data['afh']

fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, sharex=True)

ax1.plot(torque_history.T, linewidth=0.5)

ax2.plot(force_history.T,linewidth=0.5)

ax3.plot(ref_wrench_history.T,linewidth=0.5)

ax4.plot(np.array([1,1.05,1.1,1.15])*contacts_history.T,linewidth=0.5)

ax5.plot(np.array([1,1.05,1.1,1.15])*active_feet_history.T,linewidth=0.5)

plt.show()