import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()

# ax = fig.add_subplot(projection='3d')

# pointCloud = np.loadtxt("pointCloud.dat")
# depthMap = np.loadtxt("depthImage.dat")


 
# # Creating plot
# ax.scatter(pointCloud[:,0], pointCloud[:,1], pointCloud[:,2],s=0.05,c=pointCloud[:,2])
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# # ax.invert_xaxis()

# ax.view_init(20 ,20)

fig = plt.figure()
plt.ion()
ax = fig.gca(projection='3d')

pcl = np.loadtxt("points")
print (pcl)
im = ax.scatter (pcl[:,0],pcl[:,1],pcl[:,2],s=3, color="black")
ax.set_xlabel ('x')
ax.set_ylabel ('y')
ax.set_zlabel ('z')

while True:
    try:
        pcl = np.loadtxt("points")
        fig.cla ()
        im = ax.scatter (pcl[:,0],pcl[:,1],pcl[:,2],s=3, color="black")
        fig.draw ()
    except:
        pass
    
    # show plot
    plt.pause(0.2)