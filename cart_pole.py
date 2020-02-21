import gym
# Start medium artikkel
import numpy as np
import random
import math

# Initialize q-table values to 0
degIntervals = 8 # must be 8
posIntervals = 8 # must be 8
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
        if Q[state, 0] > Q[state, 1]:
            return 0
        return 1
        #return (0 if Q[state, 0] > Q[state, 1] else 1)


# Update q values
def oneDtoXY(val):
    x = val % posIntervals
    y = (val - x) // degIntervals
    return (x, y)


def updateQValues(state, action, reward):
    # TODO
    lr = 0.1
    gamma = 0.90
    x, y = oneDtoXY(state)
    up = valToState(x, min(posIntervals, y + 1))
    down = valToState(x, max(0, y - 1))
    left = valToState(max(0, x - 1), y)
    right = valToState(min(degIntervals, x + 1), y)
    Qnext = [Q[state, :], Q[up, :], Q[down, :], Q[right, :], Q[left, :]]
    Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Qnext) - Q[state, action])


def radToVal(rad):
    interval = 0;
    while True:
        if(interval*3-12)<math.degrees(rad)<(interval+1)*3-12:
            return interval
        interval+=1


def posToVal(pos):
    interval = 0;
    while True:
        if (interval * 0.6 - 2.4) < pos < (interval +1) * 0.6 -2.4:
            return interval
        interval += 1


def valToState(angval, posval):
    return angval * posIntervals + posval


# Slutt medium artikkel

env = gym.make('CartPole-v1')
results = []
episodes = 100
for i_episode in range(episodes):
    observation = env.reset()
    for t in range(100):
        if i_episode > episodes-20:
            env.render()
        angleVal = radToVal(observation[2])

        posVal = posToVal(observation[0])
        currentState = valToState(angleVal, posVal)
        action = takeAction(currentState)
        observation, reward, done, info = env.step(action)
        if done:
            results.append(t)
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
print(max(results))
env.close()

'''
Q-læring
Q(s,a)=Q(s,a)+a(R(s) + g max Q(s', a')-Q(s,a))


'''
