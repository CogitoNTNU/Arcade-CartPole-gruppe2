import gym
# Start medium artikkel
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Initialize q-table values to 0
degIntervals = 10
posIntervals = 2
tipSpeedIntervals = 40
state_size =  tipSpeedIntervals
action_size = 2
Q = np.zeros((state_size, action_size))


# Set the percent you want to explore
def takeAction(state, episode):
    epsilon = 1
    if epsilon > 0:
        epsilon -= 0.001

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
        # return (0 if Q[state, 0] > Q[state, 1] else 1)


# Update q values
'''
def oneDtoXY(val):
    x = val % posIntervals
    y = (val - x) // degIntervals
    return (x, y)


def updateQValues(state, action, reward):
    # TODO
    lr = 0.5
    gamma = 0.90
    x, y = oneDtoXY(state)
    up = valToState(x, min(posIntervals, y + 1))
    down = valToState(x, max(0, y - 1))
    left = valToState(max(0, x - 1), y)
    right = valToState(min(degIntervals, x + 1), y)
    Qnext = [Q[state, :], Q[up, :], Q[down, :], Q[right, :], Q[left, :]]
    Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Qnext) - Q[state, action])
'''

def updateQValuesBasedOnPrevAndCurrentState(prevState, prevAction, reward, curState):
    lr = 0.1
    gamma = 0.99
    Q[prevState, prevAction] = Q[prevState, prevAction] + lr * (
                reward + gamma * np.max(Q[curState, :]) - Q[prevState, prevAction])


def radToVal(rad):
    '''
    interval = 0;
    while True:
        if (interval * 3 - 12) < math.degrees(rad) < (interval + 1) * 3 - 12:
            return interval
        interval += 1
    '''
    degStart = -12
    degSlutt = 12
    bokser = np.linspace(degStart, degSlutt, degIntervals)
    for i in range(0, len(bokser)-1):
        if bokser[i] < math.degrees(rad)<bokser[i+1]:
            return i




def posToVal(pos):
    '''
    interval = 0;
    while True:
        if (interval * 0.6 - 2.4) < pos < (interval + 1) * 0.6 - 2.4:
            return interval
        interval += 1
    '''
    posStart = -2.4
    posSlutt = 2.4
    bokser = np.linspace(posStart, posSlutt, posIntervals)
    for i in range(0, len(bokser) - 1):
        if bokser[i] < pos < bokser[i + 1]:
            return i

def tipSpeedToVal(speed):
    speedStart = -3.1
    speedSlutt = 3.1
    bokser = np.linspace(speedStart, speedSlutt, tipSpeedIntervals)
    for i in range(0, len(bokser) - 1):
        if bokser[i] < speed < bokser[i + 1]:
            return i



def valToState(angval, posval, speedVal):
    # x + WIDTH * (y + DEPTH * z)
    # return angval * posIntervals + posval
    return posval + (posIntervals * (angval + degIntervals*speedVal))

def valToState2(angVal, speedVal):
    return angVal*tipSpeedIntervals +speedVal


# Slutt medium artikkel

env = gym.make('CartPole-v1')
results = []
avg = []
episodes = 1000
for i_episode in range(episodes):
    observation = env.reset()
    for t in range(200):
        if i_episode > episodes - 5:
            pass
            #env.render()
        #Verdier
        angleVal = radToVal(observation[2])
        posVal = posToVal(observation[0])
        speedVal = tipSpeedToVal(observation[3])
        prevState = valToState(angleVal, posVal, speedVal)
        prevState2 = valToState2(angleVal, speedVal)
        action = takeAction(speedVal, i_episode)
        observation, reward, done, info = env.step(action)
        if done:
            results.append(t)
            if i_episode > 20:
                avg.append(sum(results[-19:])/20)
            if i_episode%200 == 0:
                print("Episode finished after {} timesteps".format(t + 1))
            break
        #Verdier
        angleVal = radToVal(observation[2])
        posVal = posToVal(observation[0])
        speedVal2 = tipSpeedToVal(observation[3])
        newState = valToState(angleVal, posVal, speedVal)
        newState2 = valToState2(angleVal, speedVal)


        # updateQValues(new_state, action, reward)
        updateQValuesBasedOnPrevAndCurrentState(speedVal, action, reward, speedVal2)
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
plt.plot(avg)
plt.show()
env.close()

'''
Q-læring
Q(s,a)=Q(s,a)+a(R(s) + g max Q(s', a')-Q(s,a))


'''
