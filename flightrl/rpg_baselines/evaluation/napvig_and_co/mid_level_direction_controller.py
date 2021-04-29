import numpy as np

class MidLevelDirectionController:
    def __init__(self, k_omega, k, sigma):

        self.komega = k_omega
        self.gain_matrix = np.diag(k)
        self.sigma = sigma

    def genQuatrotorReferences(self, des_vel):
        vx = des_vel[0]
        vy = des_vel[1]

        # misalignment angle
        alpha = np.angle(vx+vy*1j)

        # alignment reference generator
        yaw_rate_ref = self.komega * alpha

        quadrotor_ref = np.empty(4)
        quadrotor_ref[0] = yaw_rate_ref

        # desired linear velocity: scaled by the gain matrix and by the misalignment angle
        lin_vels = np.dot(self.gain_matrix, des_vel.reshape(3,1)) * np.e**(-alpha**2 / (2*(self.sigma)**2))
        quadrotor_ref[1:] = lin_vels.reshape(-1)

        return quadrotor_ref
        
