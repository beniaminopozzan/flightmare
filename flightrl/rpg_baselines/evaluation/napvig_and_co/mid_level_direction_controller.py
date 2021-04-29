import numpy as np

class MidLevelDirectionController:
    def __init__(self):
        self.komega = 1.0
        self.kx = 3.0
        self.ky = 0.0
        self.kz = 3.0

        self.gain_matrix = np.diag([self.kx, self.ky, self.kz])

    def genQuatrotorReferences(self, des_vel):
        vx = des_vel[0]
        vy = des_vel[1]
        vz = des_vel[2]

        # misalignment angle
        alpha = np.angle(vx+vy*1j)

        # alignment reference generator
        yaw_rate_ref = self.komega * alpha

        quadrotor_ref = np.empty((4,1))
        quadrotor_ref[0,0] = yaw_rate_ref

        lin_vels = np.dot(self.gain_matrix, des_vel.reshape(3,1)) * np.e**(-alpha**2 / (2*(50)**2))
        quadrotor_ref[1:,:] = lin_vels

        return quadrotor_ref.ravel()
        
