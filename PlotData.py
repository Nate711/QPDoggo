import matplotlib.pyplot as plt
import numpy as np
import pickle

### Init graphics ###
infile = open('woofer_logs.pickle','rb')
data = pickle.load(infile)
infile.close()


fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, sharex=True)
# fig, (ax1, ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, sharex=True)

ax1.plot(data['torque_history'].T, linewidth=0.5)
ax1.set_title('torque_history')

ax2.plot(data['force_history'].T,linewidth=0.5)
ax2.set_title('force_history')

ref = ax3.plot(data['ref_wrench_history'].T,linewidth=0.5)
ax3.set_title('ref_wrench_history')
ax3.legend(ref, ('x','y','z','r','p','y'))

ax4.plot(np.array([1,1.05,1.1,1.15])*data['contacts_history'].T,linewidth=0.5)
ax4.set_title('contacts_history')

ax5.plot(np.array([1,1.05,1.1,1.15])*data['active_feet_history'].T,linewidth=0.5)
ax5.set_title('active_feet_history')

# ax6.plot(smooth_contacts_history.T,linewidth=0.5)


fig2, (ax1,ax2,ax3,ax4) = plt.subplots(nrows = 4,sharex = True)
ax1.plot(data['swing_force_history'].T, linewidth=0.5)
ax2.plot(data['swing_trajectory'].T, linewidth=0.5)
ax3.plot(data['phase_history'].T, linewidth=0.5)
ax4.plot(data['step_phase_history'].T, linewidth=0.5)

plt.show()