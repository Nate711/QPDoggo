import numpy as np
from math import cos, sin, acos


############## Quaternion Functions ##############
# see https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
# for everything and more that you want to know about quaternions
# convention for quaternions: [s v] where s is scalar, v is vector portion

def prod(a, b):
# computes the quaternion product of two input quaternions (in 4
# dimensional vector form)
	c = np.zeros((4))
	c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
	c[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
	c[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
	c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]

	return c

def exp(v):
# converts 3 element axis angle vector to a quaternions
	q = np.zeros((4));

	phi = np.linalg.norm(v)
	eps = 1e-17

	if (phi > eps):
		u = v/phi
	else:
		u = v

	q[0] = cos(phi/2)
	q[1:4] = u*sin(phi/2)

	return q

def log(q):
# convert quaternion to axis angle vector via log map
	eps = 1e-3

	v_norm = np.linalg.norm(q[1:4])

	if v_norm > eps:
		phi = 2*acos(q[0])
		u = q[1:4]/v_norm

		q_l = phi*u
	else:
		q_l = 2*q[1:4]/q[0]*(1 - v_norm)**2/(3*q[0]**2)

	return q_l

def inv(q):
# returns the inverse of input quaternion q
	q[1:4] = -q[1:4]
	return q

def vectorRotation(q, v):
# rotate a vector with a quaternion
	return prod(q, prod(v, inv(q)))

def fromVector(v):
# return the quaternion version of a vector
	q = np.zeros((4))
	q[1:4] = v

	return q

def normalize(a):
# returns normalized vector/quaternion
	return a/np.linalg.norm(a)
