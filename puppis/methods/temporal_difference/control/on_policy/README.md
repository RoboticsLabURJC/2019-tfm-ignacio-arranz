# On-policy learning: SARSA
State-Action-Reward-State-Action (SARSA) is an on-policy TD algorithm. Here we focus on state-action value instead of a state-value pair. In SARSA we update the Q value based on the following update rule:

Q(s,a) = Q(s,a) + alpha (r + gamma Q(s',a') - Q(s,a))

The steps involved in SARSA are as follows:
1. We initialize the Q values to some arbitrary values.
2. We select an action by the epsilon-greedy policy (e > 0) and move from one state to another.
3. We update the Q value previous state by following the update rule Q(s,a) = Q(s,a) + alpha(r + gamma Q(s',a) - Q(s,a)), where a' is the action selected by an epsilon-greedy policy (e > 0).

![](sarsa.gif)
