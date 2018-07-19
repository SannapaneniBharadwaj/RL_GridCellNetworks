import numpy as np
import math
import matplotlib.pyplot as plt

MU = 0.01
SIGMA = 5.90
B = 13.25
ROOM_LEN = 50
NUM_STEPS = 10000
STEP_LEN = 0.01


def sign(x):
    return x / abs(x)


def wrap(x):
    x = x % (2 * math.pi)
    return x


# for square room of size ROOM_LEN x ROOM_LEN
def minDistAngle(position, theta):
    if position[0] < 2:
        dWall = position[0]
        aWall = theta + math.pi
    elif position[1] < 2:
        dWall = position[1]
        aWall = theta + 3 * math.pi / 2
    elif position[0] > 48:
        dWall = 50 - position[0]
        aWall = theta
    elif position[1] > 48:
        dWall = 50 - position[1]
        aWall = theta + math.pi/2
    else:
        # dWall = 3 causes
        dWall = 3
        aWall = 0

    aWall = wrap(aWall)

    return dWall, aWall


x_pos = []
y_pos = []
thetas = []
velocities = []
omegas = []

random_w = np.random.normal(MU, SIGMA, NUM_STEPS)
random_v = np.random.rayleigh(B, NUM_STEPS)

velocity = np.array([20., 20.])
omega = 0.  # angular velocity
position = np.array([20., 25.])
theta = 0.  # angular position

# store in list
x_pos.append(position[0])
y_pos.append(position[1])
thetas.append(theta)
velocities.append(velocity)
omegas.append(theta)

for i in range(NUM_STEPS):
    dWall, aWall = minDistAngle(position, theta)
    #print(dWall, aWall)

    if dWall < 2 and 0 < aWall and aWall < math.pi:
        # turn
        theta = sign(aWall) * (math.pi - abs(aWall)) + random_w[i]
        theta = wrap(theta)
        # slow down
        velocity += -.5 * (velocity - 5)
    else:
        speed = random_v[i]
        omega = random_w[i]

    # update position, angle
    velocity = speed * np.array([math.sin(theta), math.cos(theta)])
    position += velocity * STEP_LEN
    theta += omega * STEP_LEN
    theta = wrap(theta)

    #print('i', i)
    #print('position', position)
    #print('theta', theta)

    # store in list
    x_pos.append(position[0])
    y_pos.append(position[1])
    thetas.append(theta)
    velocities.append(velocity)
    omegas.append(theta)

#print(min(x_pos))
#print(max(x_pos))
#print(min(y_pos))
#print(max(y_pos))
plt.plot(x_pos, y_pos)
plt.show()