# Monte Carlo Prediction
In Monte Carlo prediction we approximate the value function by taking the mean return instead of the expected return. We can estimate the value function of any given policy.

Steps:
1. First, we initialize a random value to our value function.
2. Then we initialize an empty list called a return to store our returns.
3. Then for each state in the episode, we calculate the return.
4. Next, we append the return to our return list.
5. Finally, we take the average of return as our value function.

There are two types of Monte Carlo prediction:
* **First visit Monte Carlo**: in the Monte Carlo methods we approximate the value function by taking the average return. But in the first Monte Carlo method, we average the return only the first time the estate is visited in an episode. The policy is given, it calcuate the value function.
* **Every visit Monte Carlo**: we average the return every time the state is visited in an episode. Some policy is given, it finds the optimal policy.
 