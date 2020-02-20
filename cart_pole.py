import gym
# Start medium artikkel
import numpy as np
import random
import math

# Initialize q-table values to 0
degIntervals = 9
posIntervals = 8
state_size = degIntervals * posIntervals
action_size = 2
Q = np.zeros((state_size, action_size))


# Set the percent you want to explore
def takeAction(state):
    epsilon = 0.2
    if random.uniform(0, 1) < epsilon:
        """
        Explore: select a random action
        """
        return env.action_space.sample()
    else:
        """
        Exploit: select the action with max value (future reward)
        """
        return 0 if Q[state, 0] > Q[state, 1] else 1


# Update q values
def oneDtoXY(val):
    x  =val%degIntervals
    y = (val-x)/posIntervals
    return(x,y)
def updateQValues(state, action, reward):
    #TODO
    lr = 0.5
    gamma = 0.9
    x,y = oneDtoXY(state)
    up = valToState(x, y)
    down = valToState(x,max(y, y-1))
    left = 0
    right = 0
    Qnext = [Q[state,:], Q[up, :], Q[down,:],Q[right,:], Q[left,:] ]
    Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Qnext) - Q[state, action])


def radToVal(rad):
    ranges = [deg for deg in range(-12, 12, degIntervals)]
    degrees = math.degrees(rad)
    for i in range(0, len(ranges)):
        if ranges[i] < degrees < ranges[i + 1]:
            return i
def posToVal(pos):
    ranges = [pos for pos in range(-5, 5, posIntervals)]
    for i in range(0, len(ranges)):
        if ranges[i] < pos < ranges[i + 1]:
            return i
def valToState(angval, posval):
    return angval*degIntervals+posval



# Slutt medium artikkel

env = gym.make('CartPole-v1')

for i_episode in range(2000):
    observation = env.reset()
    for t in range(100):
        env.render()
        angleVal =radToVal(observation[2])
        posVal = posToVal(observation[0])
        currentState = valToState(angleVal, posVal)
        action = takeAction(currentState)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        angleVal = radToVal(observation[2])
        posVal = posToVal(observation[0])
        new_state = valToState(angleVal, posVal)
        updateQValues(new_state, action, reward)
        '''
         Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
        reward float
        +1 per time step
        done bool
        info dict
        '''

print(Q)
env.close()

'''
Q-læring
Q(s,a)=Q(s,a)+a(R(s) + g max Q(s', a')-Q(s,a))


'''
