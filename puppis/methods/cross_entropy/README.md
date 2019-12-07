# Cross-Entropy
1. Play N number of episodes using our current model and environment.
2. Calculate the total reward for every episode and decide on a reward boundary. Usually, we use some percentile of
all rewards, such 50 or 70.
3. Throw away all episodes with a reward below the boundary.
4. Train a nn on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
5. Repeat from 1 until achieve your goal.
  
![](cross_entropy.gif)
