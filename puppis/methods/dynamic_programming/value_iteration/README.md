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

# Value Iteration
We start off with a random value function, so we look for a new improved value function in iterative fashion until we find the optimal policy function. Once we find the optimal value function, we can easily derive an optimal policy from it.

The steps involved in the value iteration are as follows:
1. First, we initialize the random value function (random value for each state).
2. Then we compute the Q function for all states action pairs of Q(s,a).
3. Then we update our value function with the max value from Q(s,a).
4. We repeat these steps until the changes in the value function is very small.

##### 1.Initialize the random value function
```python
value_table = np.zeros(gym_env.observation_space.n)
n_iterations = 10000
```

##### 2.Compute Q function
```python

updated_value_table = np.copy(value_table)
for state in range(gym_env.observation_space.n):
    Q_value = []
    for action in range(gym_env.action_space.n):
        next_states_rewards = []
        for next_sr in gym_env.env.P[state][action]:
            trans_prob, next_state, reward_prob, _ = next_sr
            next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

        Q_value.append(np.sum(next_states_rewards))
```

##### 3.Update value function wit the max value from Q(s,a)
```python
value_table[state] = max(Q_value)
```

##### 4.Repeat steps until changes in the value function is very small
```python
if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
    print('Value-iteration converged at iteration# {}'.format((i+1)))
```

### Results

```python
if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    optimal_value_function = value_itearation(env, 1.0)
    optimal_policy = extract_policy(env, optimal_value_function, 1.0)
    print(optimal_policy)
```

```python
Value-iteration converged at iteration# 1373
...
Value-iteration converged at iteration# 10000
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