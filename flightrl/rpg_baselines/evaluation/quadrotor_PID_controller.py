import numpy as np

from scipy.spatial.transform import Rotation as R

class QuadrotorPIDcontroller():
    def __init__(self):
        # allocation matrix
        self.B =  np.empty((4,4))
        self.B[0,:] = [1, 1, 1, 1]
        self.B[1,:] = 0.17 * np.sqrt(0.5)* np.array([1, -1, -1, 1])
        self.B[2,:] = 0.17 * np.sqrt(0.5)* np.array([-1, -1, 1, 1])
        self.B[3,:] = 0.016 * np.array([1, -1, 1, -1])
        self.B_inv = np.linalg.inv(self.B)
        
        # x and y rate, body frame, controllers. Outputs: desired linear acceleration
        self.x_rate_controller = PID(k_p=5, u_max=9.81*np.pi/4, u_min=-9.81*np.pi/4)
        self.y_rate_controller = PID(k_p=5, u_max=9.81*np.pi/4, u_min=-9.81*np.pi/4)

        # pitch rate controller
        self.pitch_rate_controller = PID(k_p=1, u_max=2.0, u_min=-2.0)
        # pitch controller
        self.pitch_controller = PID(k_p=5, u_max=2, u_min=-2)
        # roll rate controller
        self.roll_rate_controller = PID(k_p=1, u_max=2.0, u_min=-2.0)
        # roll controller
        self.roll_controller = PID(k_p=5, u_max=2, u_min=-2)

        # yaw rate controller
        self.yaw_rate_controller = PID(k_p=1,u_max=0.05,u_min=-0.05)
        # yaw controller
        self.yaw_controller = PID(k_p=2, u_max=0.5, u_min=-0.5)

        # altitude rate controller
        self.altitude_rate_controller = PID(k_p=10, k_i=2, k_d=1)

        #self.f = open('pitch.dat','a')
        #self.q = open('Rmat.dat','a')

    # def retrieve_yaw(self,obs):
    #     number_of_half_turns = self.yaw_angle // (np.pi)
    #     yaw = obs[3]
    #     delta = (yaw + (np.pi)*number_of_half_turns) - self.yaw_angle
    #     if delta > np.pi/2:
    #         delta = delta - (np.pi)
    #     elif delta < -np.pi/2:
    #         delta = np.pi + delta
    #     self.yaw_angle = self.yaw_angle + delta

    # def retrieve_pitch(self,obs):
    #     number_of_half_turns = self.pitch_angle // (np.pi)
    #     pitch = obs[4]
    #     delta = (pitch + (np.pi)*number_of_half_turns) - self.pitch_angle
    #     if delta > np.pi/2:
    #         delta = delta - (np.pi)
    #     elif delta < -np.pi/2:
    #         delta = np.pi + delta
    #     self.pitch_angle = self.pitch_angle + delta

    
    def apply(self, obs, ref):
        #self.retrieve_yaw(obs)
        #self.retrieve_pitch(obs)

        yaw_ref = ref[0]
        x_rate_ref = ref[1]
        y_rate_ref = ref[2]
        elev_rate_ref = ref[3]

        r=R.from_euler('ZYX',obs[3:6])
        angles = r.as_euler('ZYX')
        yaw_angle = angles[0]
        pitch_angle = angles[1]
        roll_angle = angles[2]
        x_rate = np.cos(yaw_angle)*obs[6] + np.sin(yaw_angle)*obs[7]
        y_rate = -np.sin(yaw_angle)*obs[6] + np.cos(yaw_angle)*obs[7]
        elev_rate = obs[8]
        roll_rate = obs[9]
        pitch_rate = obs[10]
        yaw_rate = obs[11]

        elev_rate_e = elev_rate_ref - elev_rate
        thrust = self.altitude_rate_controller.apply(elev_rate_e) + 0.73*9.81
        
        x_rate_e = x_rate_ref - x_rate
        pitch_ref = 0.73/thrust * self.x_rate_controller.apply(x_rate_e)
        #pitch_ref = 1/9.81 * self.x_rate_controller.apply(x_rate_e)

        y_rate_e = y_rate_ref - y_rate
        roll_ref = -0.73/thrust * self.y_rate_controller.apply(y_rate_e)

        pitch_e = pitch_ref - pitch_angle
        pitch_rate_ref = self.pitch_controller.apply(pitch_e)
        pitch_rate_e = pitch_rate_ref - pitch_rate        
        tau_y = self.pitch_rate_controller.apply(pitch_rate_e)


        # print("thrust={2:>{0}.{1}f}".format(10,3,thrust))
        # print("    x_rate_ref={2:>{0}.{1}f};        x_rate={3:>{0}.{1}f}".format(6,3,x_rate_ref,x_rate))
        # print("     pitch_ref={2:>{0}.{1}f};   pitch_angle={3:>{0}.{1}f}".format(6,3,pitch_ref,pitch_angle))
        # print("pitch_rate_ref={2:>{0}.{1}f};     pitch_rate={3:>{0}.{1}f};   tau_y={4:>{0}.{1}f}".format(6,3,pitch_rate_ref,pitch_rate,tau_y))

        roll_e = roll_ref - roll_angle
        roll_rate_ref = self.roll_controller.apply(roll_e)
        roll_rate_e = roll_rate_ref - roll_rate        
        tau_x = self.roll_rate_controller.apply(roll_rate_e)

        yaw_e = yaw_ref - yaw_angle
        yaw_rate_ref = self.yaw_controller.apply(yaw_e)
        yaw_rate_e = yaw_rate_ref - yaw_rate
        tau_z = self.yaw_rate_controller.apply(yaw_rate_e)

        # print("thrust={2:>{0}.{1}f}".format(10,3,thrust))
        # print("     yaw_ref={2:>{0}.{1}f};   yaw_angle={3:>{0}.{1}f}".format(6,3,yaw_ref,yaw_angle))
        # print("yaw_rate_ref={2:>{0}.{1}f};    yaw_rate={3:>{0}.{1}f};   tau_z={4:>{0}.{1}f}".format(6,3,yaw_rate_ref,yaw_rate,tau_z))


        # print("  yaw={2:>{0}.{1}f};   yaw_rate={3:>{0}.{1}f}".format(6,3,yaw_angle,yaw_rate))
        # print("pitch={2:>{0}.{1}f}; pitch_rate={3:>{0}.{1}f}".format(6,3,pitch_angle,pitch_rate))
        # print(" roll={2:>{0}.{1}f};  roll_rate={3:>{0}.{1}f}".format(6,3,roll_angle,roll_rate))

        # print("x rate={2:>{0}.{1}f}; y rate={3:>{0}.{1}f}".format(6,3,x_rate,y_rate))

        #np.savetxt(self.f,np.array([obs[4]]))

        
        #np.savetxt(self.q,r.reshape((1,9),order='F'))
        

        return np.dot(self.B_inv, np.array([[thrust,tau_x,tau_y,tau_z]]).transpose())



class PID():
    def __init__(self, T_s=0.02, k_p=0.0, k_i=0.0, k_d=0.0, T_f=0.0, k_awu=0.0, u_max=np.inf, u_min=-np.inf):
        # PID params initialization
        self.T_s = T_s
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.T_f = T_s
        self.k_awu = k_awu
        self.u_max = u_max
        self.u_min = u_min
        # integrator output
        self.u_i = 0.0
        # past input value for the discrete derivative
        self.e_old = 0.0
        # past derivator output
        self.u_d = 0.0

        
    
    def apply(self, e):
        # proportional component
        u_p = self.k_p * e
        # derivative component
        d_e = (e-self.e_old)
        self.u_d = 1/(self.T_f+self.T_s) * (self.k_d * d_e + self.T_s*self.u_d)
        # integral component
        self.u_i = self.u_i + self.k_i * self.T_s * e
        # output NON saturated
        u = u_p + self.u_i + self.u_d
        # apply satuation
        u_sat = np.minimum(self.u_max, np.maximum(self.u_min,u))
        sat_err = u - u_sat
        self.u_i = self.u_i - self.k_awu * self.T_s * sat_err
        return u_sat

