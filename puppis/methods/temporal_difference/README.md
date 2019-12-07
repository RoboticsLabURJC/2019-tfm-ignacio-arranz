# Temporal Difference Learning
The algorithm takes the benefits of both the Monte Carlo method and dynamic programming (DP) into account. It doesn't require model dynamics (Monte Carlo) and it doesn't need to wait until the end of the episode to make an estimate of the value function (DP).
Instead, it approximates the current estimate based on the previously learned estimated, which is also called bootstrapping. In Monte Carlo methods we only estimate  only at the end of the episode.

* TD prediction
* TD control

## TD prediction
TD prediction try to predict the state values. We update the value of a previous state by current state. TD learning using a TD update rule for updating the value of a state.

V(s) = V(s) + alpha(r + gamma * V(s') - V(s))

Where:
* V(s): value of previous state.
* alpha: learning rate.
* r: reward.
* gamma: discount rate.
* V(s'): value of actual state.

It is actually the differentce between the actual reward (r + gamma * V(s')) and the expected reward (V(s)) multiplied by the learning rate alpha. Since we take the difference between the actual and predicted value as (r + gamma * V(s') - V(s)), it is actually an error. We can call it a TD error. Over several iterations, we will try to minimize this error.

The steps involved in the TD-prediction algorithm are as follows:
1. We initialize V(S) to 0 or som e arbitrary values.
2. We begin the episode the for every step in the episode, we perform an action A in the state S and receive a reward R and move to the next state (s').
3. We update the value of the previous state using the TD update rule.
4. We repeat steps 3 and 4 until we reach the terminal state.

