import numpy as np
import gym
import random
import math

env1 =["SHFF", "FFFH", "FHFH", "HFFG"]
env2= ["SFFFFF", "FFFHFF", "FHFHHH", "HFFFFG"]
env3 = ['SFFHFFHH', 'HFFFFFHF', 'HFFHHFHH', 'HFHHHFFF', 'HFHHFHFF', 'FFFFFFFH', 'FHHFHFHH', 'FHHFHFFG'] 

selectedEnv = env2
env = gym.make('FrozenLake-v1', desc=selectedEnv, render_mode="human", is_slippery = False)
env.reset()
env.render()

# change-able parameters:
discount_factor = 0.99
delta_threshold = 0.00001
epsilon = 1

def value_iteration(env, gamma=0.9, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function for all states as 0
    V = np.zeros(num_states)
    
    # Learning the policy here
    while True:
        delta = 0
        for state in range(num_states):
            v = V[state] #value of the current state
            # Applying the Bellman equation to update the value of the state
            temp_values = np.zeros(num_actions) 
            for action in range(num_actions): #for each action
                for prob, next_state, reward, _ in env.P[state][action]: #for each possible transition
                    temp_values[action] += prob * (reward + gamma * V[next_state]) #summation of the expected value of the next state
             #Updating the value of the state by choosing action that maximizes the expected value of the next state
            V[state] = np.max(temp_values)
            delta = max(delta, np.abs(v - V[state])) #calculating the change in value of the state
        if delta < epsilon: #if the change in value of the state is less than the threshold the optimal plocy is learnt
            break

    #Code for extracting optimal policy from the value function

    policy = np.zeros(num_states, dtype=int)   #initializing the policy
    for state in range(num_states): #for each state
        temp_values = np.zeros(num_actions) 
        for action in range(num_actions):  #for each action
            for prob, next_state, reward,_ in env.P[state][action]: #for each possible transition
                temp_values[action] += prob * (reward + gamma * V[next_state]) #summation of the expected value of the next state
        policy[state] = np.argmax(temp_values) #selecting the action that maximizes the expected value of the next state
    
    return policy, V

# Run value iteration
policy, V = value_iteration(env, gamma=discount_factor, epsilon=delta_threshold)

# Print results
print("Optimal Value Function:")
print(V.reshape(len(selectedEnv), len(selectedEnv[0])))

print("\nOptimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(len(selectedEnv), len(selectedEnv[0])))

# resetting the environment and executing the policy
state = env.reset()
state = state[0]
step = 0
done = False
print(state)

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
    action =policy[state]
    new_state, reward, done, truncated, info = env.step(action) #information after taking the action

    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state

env.close()