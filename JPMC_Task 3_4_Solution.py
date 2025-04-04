import pandas as pd
import numpy as np

def log_likelihood(n, k):
   
    if n == 0 or k == 0 or k == n:
        return 0
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)

def find_optimal_buckets(fico_scores, defaults, num_buckets):
    
    fico_scores = np.sort(fico_scores)  
    n = len(fico_scores)
    dp = np.zeros((num_buckets + 1, n)) - np.inf 
    dp[0, :] = 0 

    for i in range(1, num_buckets + 1):
        for j in range(i - 1, n):
            cumulative_defaults = np.cumsum(defaults[:j + 1])
            cumulative_observations = np.arange(1, j + 2)
            for k in range(i - 1, j):
                bucket_defaults = cumulative_defaults[j] - cumulative_defaults[k]
                bucket_observations = j - k
                current_likelihood = dp[i - 1, k] + log_likelihood(bucket_observations, bucket_defaults)
                if current_likelihood > dp[i, j]:
                    dp[i, j] = current_likelihood
                    dp[i, j + 1] = k  

   
    boundaries = []
    current_index = n - 1
    for i in range(num_buckets - 1, 0, -1):
        boundaries.append(fico_scores[int(dp[i, current_index + 1])])
        current_index = int(dp[i, current_index + 1])

    return sorted(boundaries)


num_buckets = 5 
optimal_boundaries = find_optimal_buckets(fico_scores, defaults, num_buckets)

print("Optimal Bucket Boundaries:", optimal_boundaries)






def get_rating(fico_score, boundaries):
    for i, boundary in enumerate(boundaries):
        if fico_score <= boundary:
            return i + 1  
    return num_buckets





df['fico_rating'] = df['fico_score'].apply(lambda x: get_rating(x, optimal_boundaries))

