# Environment

|   Map    |
|----------|
| S F F F  |
| F H F H  |
| F F F H  |
| H F F G  |

Where:
* S: starting point (safe).
* F: frozen surface (safe).
* H: hole, fall to your doom.
* G: goal.

States: 16

|Action|Value|
|------|-----|
| LEFT |  0  |
| DOWN |  1  |
| RIGHT|  2  |
|  UP  |  3  |



[Fronzen Lake (OpenAI)](https://gym.openai.com/envs/FrozenLake-v0/)

We can model our problem into MDP:
* States: set of states. Here we have 16 (each little square box in the grid).
* Actions: set of all possible actions (right, left, down and up).
* Transition probabilities: the probability of moving from one state (F) to another state (H) by performing an action a.
* Reward probabilities: this is the probability of receiving a reward while moving from one state (F) to another state (H) by performing an action a.


Solving the MDP implies finding the optimal policies:
* Policy function: specifies what action to perform in each state.
* Value function: specifies how good a state is.
* Q function: specifies how good an action is in a particular state.

We represent the value function and Q function using the Bellman Optimality equation.

# Policy Iteration
Unlike value iteration, in policy iteration we start with the random policy, then we find the value function of that policy; if the value is not optimal then we find the new improved policy. We repeat this process until we find the optimal policy.

There are wo steps in policy iteration:
* Policy evaluation: evaluating the value function of a randomly estimated policy.
* Policy improvement: upon evaluation the value function, if it is not optimal, we find a new improved policy.

The steps involved in the policy iteration are as follow:
1. First, we initialize some random policy.
2. Then we find the value function for that random policy and evaluate to check if it is optimal which is called policy evaluation.
3. If it not optimal, we find a new improved policy, which is called policy improvement.
4. We repeat these steps until we find an optimal policy.


##### 1.Initialize the random policy
```python
random_policy = np.zeros(gym_env.observation_space.n)
iterations = 200000
```

##### 2.Find the value function for random policy
```python
updated_value_table = np.copy(value_table)
    for state in range(env.env.nS):
        action = policy[state]

        value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                  for trans_prob, next_state, reward_prob, _ in gym_env.env.P[state][action]])
```

##### 3.Improve policy
```python
policy = np.zeros(gym_env.observation_space.n)
for state in range(gym_env.observation_space.n):
    Q_table = np.zeros(gym_env.action_space.n)
    for action in range(gym_env.action_space.n):
        for next_sr in gym_env.env.P[state][action]:
            trans_prob, next_state, reward_prob, _ = next_sr
            Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

    policy[state] = np.argmax(Q_table)

return policy
```

##### 4.Repeat these steps until we find an optimal policy
```python
if np.all(random_policy == new_policy):
    print('Policy iteration converged at step {}.'.format(i+1))
    break
random_policy = new_policy
```

### Results

```python
if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    result = policy_iteration(env, 1.0)
    print(result)
```

```python
Policy iteration converged at step 7.
[0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
```
##### Optimal policy

|          Map             |
|--------------------------|
| S     F       F       F  |
| F     H       F       H  |
| F     F       F       H  |
| H     F       F       G  |

Actions to be performed in each state.
###### Gamma=1
```python
[LEFT   UP      UP      UP 
 LEFT   LEFT    LEFT    LEFT 
 UP     DOWN    LEFT    LEFT 
 LEFT   RIGHT   DOWN    LEFT]
```
###### Gamma=0.5
```python
[DOWN   UP      RIGHT      UP 
 LEFT   LEFT    LEFT    LEFT 
 UP     DOWN    LEFT    LEFT 
 LEFT   RIGHT   DOWN    LEFT]
```
###### Gamma=0
```python
[LEFT   LEFT   LEFT    LEFT 
 LEFT   LEFT   LEFT    LEFT 
 LEFT   LEFT   LEFT    LEFT 
 LEFT   LEFT   DOWN    LEFT]
```