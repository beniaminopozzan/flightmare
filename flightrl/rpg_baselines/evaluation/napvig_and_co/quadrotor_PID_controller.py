import numpy as np

from scipy.spatial.transform import Rotation as R

class QuadrotorPIDcontroller():
    def __init__(
        self,
        control_params,
    ):
        # allocation matrix
        kappa = control_params['kappa']
        arm_l = control_params['arm_l']
        self.B =  np.empty((4,4))
        self.B[0,:] = [1, 1, 1, 1]
        self.B[1,:] = arm_l * np.sqrt(0.5)* np.array([1, -1, -1, 1])
        self.B[2,:] = arm_l * np.sqrt(0.5)* np.array([-1, -1, 1, 1])
        self.B[3,:] = kappa * np.array([1, -1, 1, -1])
        self.B_inv = np.linalg.inv(self.B)

        
        self.mass = control_params['mass']
        self.g = control_params['g']

        Ts = control_params['T_s']
        
        # x and y rate, body frame, controllers. Outputs: desired linear thrusts [N]
        # that is then converted to desired angle
        self.x_rate_controller = PID(T_s=Ts,**control_params['x_rate_c_params'])
        self.y_rate_controller = PID(T_s=Ts,**control_params['y_rate_c_params'])
        # pitch and roll saturators [rad]
        self.pitch_angle_saturator = PID(T_s=Ts,**control_params['pitch_angle_saturator'])
        self.roll_angle_saturator = PID(T_s=Ts,**control_params['roll_angle_saturator'])

        # pitch controller. Output: desired pitch rates [rad/s]
        self.pitch_controller = PID(T_s=Ts,**control_params['pitch_controller'])
        # pitch rate controller. Output: desired pitch torque [Nm]
        self.pitch_rate_controller = PID(T_s=Ts,**control_params['pitch_rate_controller'])
        # roll controller. Output: desired roll rate [rad/s]
        self.roll_controller = PID(T_s=Ts,**control_params['roll_controller'])
        # roll rate controller. Output: desired roll torque [Nm]
        self.roll_rate_controller = PID(T_s=Ts,**control_params['roll_rate_controller'])

        # yaw rate controller. Output: desired yaw torque [Nm]
        self.yaw_rate_controller = PID(T_s=Ts,**control_params['yaw_rate_controller'])

        # altitude rate controller. Output: desired thrust [N] (gravity not compensated)
        self.altitude_rate_controller = PID(T_s=Ts,**control_params['altitude_rate_controller'])
    
    def reset(self):
        self.x_rate_controller.reset()
        self.y_rate_controller.reset()
        self.pitch_angle_saturator.reset()
        self.roll_angle_saturator.reset()
        self.pitch_controller.reset()
        self.pitch_rate_controller.reset()
        self.roll_controller.reset()
        self.roll_rate_controller.reset()
        self.yaw_rate_controller.reset()
        self.altitude_rate_controller.reset()
    
    def apply(self, obs, ref):

        # retrieve the references
        yaw_rate_ref = ref[0]
        x_rate_ref = ref[1]
        y_rate_ref = ref[2]
        elev_rate_ref = ref[3]
        # retrive all the data from the observation
        r=R.from_euler('ZYX',obs[3:6])
        angles = r.as_euler('ZYX')
        yaw_angle = angles[0]
        pitch_angle = angles[1]
        roll_angle = angles[2]
        x_rate = np.cos(yaw_angle)*obs[6] + np.sin(yaw_angle)*obs[7]    #"body" (yaw compensated) frame
        y_rate = -np.sin(yaw_angle)*obs[6] + np.cos(yaw_angle)*obs[7]   #"body" (yaw compensated) frame
        elev_rate = obs[8]
        roll_rate = obs[9]
        pitch_rate = obs[10]
        yaw_rate = obs[11]

        # apply elevation rate controller
        elev_rate_e = elev_rate_ref - elev_rate
        thrust = self.altitude_rate_controller.apply(elev_rate_e) + self.mass*self.g
        
        # apply linear velocities controllers
        x_rate_e = x_rate_ref - x_rate
        pitch_ref = self.mass/thrust * self.x_rate_controller.apply(x_rate_e)
        pitch_ref = self.pitch_angle_saturator.apply(pitch_ref)
        y_rate_e = y_rate_ref - y_rate
        roll_ref = -self.mass/thrust * self.y_rate_controller.apply(y_rate_e)
        roll_ref = self.roll_angle_saturator.apply(roll_ref)

        # apply pitch controllers (angle and rate)
        pitch_e = pitch_ref - pitch_angle
        pitch_rate_ref = self.pitch_controller.apply(pitch_e)
        pitch_rate_e = pitch_rate_ref - pitch_rate        
        tau_y = self.pitch_rate_controller.apply(pitch_rate_e)

        # apply roll controllers (angle and rate)
        roll_e = roll_ref - roll_angle
        roll_rate_ref = self.roll_controller.apply(roll_e)
        roll_rate_e = roll_rate_ref - roll_rate        
        tau_x = self.roll_rate_controller.apply(roll_rate_e)

        # apply yaw rate ontroller 
        yaw_rate_e = yaw_rate_ref - yaw_rate
        tau_z = self.yaw_rate_controller.apply(yaw_rate_e)

        # print("thrust={2:>{0}.{1}f}".format(10,3,thrust))
        # print("     yaw_ref={2:>{0}.{1}f};   yaw_angle={3:>{0}.{1}f}".format(6,3,yaw_ref,yaw_angle))
        # print("yaw_rate_ref={2:>{0}.{1}f};    yaw_rate={3:>{0}.{1}f};   tau_z={4:>{0}.{1}f}".format(6,3,yaw_rate_ref,yaw_rate,tau_z))


        # print("  yaw={2:>{0}.{1}f};   yaw_rate={3:>{0}.{1}f}".format(6,3,yaw_angle,yaw_rate))
        # print("pitch={2:>{0}.{1}f}; pitch_rate={3:>{0}.{1}f}".format(6,3,pitch_angle,pitch_rate))
        # print(" roll={2:>{0}.{1}f};  roll_rate={3:>{0}.{1}f}".format(6,3,roll_angle,roll_rate))

        # print("x rate={2:>{0}.{1}f}; y rate={3:>{0}.{1}f}".format(6,3,x_rate,y_rate))

        # print("thrust={2:>{0}.{1}f}".format(10,3,thrust))
        # print("    x_rate_ref={2:>{0}.{1}f};        x_rate={3:>{0}.{1}f}".format(6,3,x_rate_ref,x_rate))
        # print("     pitch_ref={2:>{0}.{1}f};   pitch_angle={3:>{0}.{1}f}".format(6,3,pitch_ref,pitch_angle))
        # print("pitch_rate_ref={2:>{0}.{1}f};     pitch_rate={3:>{0}.{1}f};   tau_y={4:>{0}.{1}f}".format(6,3,pitch_rate_ref,pitch_rate,tau_y))

        #np.savetxt(self.f,np.array([obs[4]]))

        
        #np.savetxt(self.q,r.reshape((1,9),order='F'))
        

        return ( np.dot(self.B_inv, np.array([[thrust,tau_x,tau_y,tau_z]]).transpose()) - self.mass*self.g/4) / (self.mass*self.g/2)
    
    ### unused methods    
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
        
    def reset(self):
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

