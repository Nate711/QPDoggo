import numpy as np

# Robot joint limits
max_revolute_torque = 8
max_prismatic_force = 133

# Robot geometry
WOOFER_LEG_FB = 0.23					# front-back distance from center line to leg axis
WOOFER_LEG_LR = 0.175 					# left-right distance from center line to leg plane
WOOFER_LEG_L  = 0.32
WOOFER_ABDUCTION_OFFSET = 0				# distance from abduction axis to leg

# Robot inertia params
WOOFER_MASS = 8.0 # kg

(WOOFER_L,WOOFER_W,WOOFER_T) = (0.66, 0.176, 0.092)
Ix = WOOFER_MASS/12 * (WOOFER_W**2 + WOOFER_T**2)
Iy = WOOFER_MASS/12 * (WOOFER_L**2 + WOOFER_T**2)
Iz = WOOFER_MASS/12 * (WOOFER_L**2 + WOOFER_W**2)
WOOFER_INERTIA = np.zeros((3,3))
WOOFER_INERTIA[0,0] = Ix
WOOFER_INERTIA[1,1] = Iy
WOOFER_INERTIA[2,2] = Iz