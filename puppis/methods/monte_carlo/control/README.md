# Monte Carlo Control
In Monte Carlo control we will see how to optimize the value function, that is, how to make the value funciton more accurate than the estimation. In the control methods, we follow a new type of iteration called generalized policy iteration, where policy evaluation and policy improvement interact with each other. It basically runs as a loop between policy evaluation and improvement, that is, the policy is always improved with respect to the value function, and the value function is always improved according to the policy. It keeps on doing this. When there is no change, then we can say that the policy and value function have attained convergence, that is, we found the optimal value function and optimal policy.

## Monte Carlo exploration starts
DP methods estimate state values, instead, we focus on action values. State values alore are suffcient when we know the model of the environment. Estimating an action value is more intuitive than a state value because state value vary depending on the policy we choose. The value of the state depends on the policy we choose. So it is more important to estimate the value of an action of the value of the state.
The Q function Q(s,a) is used for determining how good an action is in a particular state.


How can we know about the state-action value if we haven't been in that state? If we don't explore all the states with all possible actions, we might probably miss out the good rewards. For us to know which is the best action, we have to explore all possible actions in each state to find the optimal value. How can we do this?

Monte Carlo exploring starts, which implies that for each episode we start with a random state as an initial state and perform an action. So, if we have a large number of episodes, we could possibly cover all the states with all possible actions. It is also called **MC-ES** algorithm.

MC-ES algorithm:
* We first initialize Q function and policy with some random values and also we initialize a return to an empty list.
* Then we start the episode with our randomly initialized policy.
* Then we calculate the return for all the unique state-action pair because the same state action pair occurs in an episode multiple times and there is no point having redundant information.
* Then we take an average of the returns in the return list and assign that value to our Q function.
* Finally, we will select an optimal policy for a state, choosing an action that has the maximum Q(s,a) for that state.
* We repeat this whole process forever or for a large number of episodes so that we can cover all different states and action pairs.

## On-policy Monte Carlo control
**greedy algorithm**: picks up the best choice available at that moment, although that choice might no be optimal.

**epsilon greedy policy**: all actions are tried with a non-zero probability (epsilon). With a probability epsilon, we explore different actions randomly and with a probability 1-epsilon we choose an action that has maximum value. So instead of just exploiting the best action all the time, with probability epsilon, we explore different actions randomly.

Steps:
1. We initialize random policy and a random Q function.
2. Then we initialize a list called return for storing the returns.
3. We generate an episode using the random policy pi.
4. We store the return of every state action pair occurring in the episode to the return list.
5. Then we take an average of the returns in the return list and assign that value to the Q function.
6. Now the probability of selecting an action a in the state s will be decided by epsilon.
7. If the probability is 1-epsilon we pick up the action which has the maximal Q value.
8. If the probability is epsilon, we explore for different actions.

  
## Off-policy Monte Carlo control
We have two policies: one is a behaviour policy and another is a target policy. In the off-policy method, agents follow one policy but in the meantime, it tries to learn and improve a different policy.
The policy an agent follows is called a behaviour policy and the policy an agent tries to evaluate and improve is called a target policy. The behaviour policy explores all possible states and actions and that is why a behaviour policy is called a soft policy, whereas a target policy is said to be a greedy policy.

Our goal is to estimate the Q function for the target policy pi, but our agents behave using a completely different policy called behaviour policy (mu). We can estimate the value of pi by using the common episodes that took place in mu. We use a technique called importance sampling. It is a technique for estimating values from one distribution given samples from another.

Importance sampling is of two types:
* Ordinary importance sample.
* Weighted importance sample.

In ordinary importance sampling, we basically take the radio of returns obtained by the behavior policy and target policy, whereas in weighted importance sampling we take the weighted average and C is the cumulative sum of weights.

Steps:
1. We initialize Q(s,a) to radom values and C(s,a) to 0 and weight w as 1.
2. Then we choose the target policy, which is a greedy policy. This means it will pick up the policy which has maximum value from the Q table.
3. We select our behaviour policy. A behavior policy is not greedy and it can select any state-action pair.
4. Then we begin our episode and perform an action a in the state s according to our behavior policy and store the reward. We repeat this until the end of the episode.
5. Now, for each state in the episode, we do the following:
    1. We will calculate return G. We know that the return is the sum of discounted rewards: G = discount_factor * G + reward.
    2. Then we update C(s,a) as C(s,a) = C(s,a) + w
    3. We update Q(s,a): Q(s,a)=Q(s,a) + w/C(s,a) * (G - Q(s,a))
    4. We updated the value of w: w = w * 1/behavior_policy