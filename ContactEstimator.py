import WooferDynamics 		

class ContactEstimator():
	"""
	Interface for generic foot contact estimator
	"""
	def update(self,sim):
		"""
		Grab data from the mujoco sim and 
		return which feet are in contact
		"""
		raise NotImplementedError
class MuJoCoContactEstimator(ContactEstimator):
	"""
	Use Mujoco contact data directly
	"""
	def __init__(self):
		pass
	def update(self,sim):
		"""
		Grab the feet contacts directly from MuJoCo sim, aka, cheat.
		"""
		contacts = WooferDynamics.FeetContacts(sim)
		return contacts