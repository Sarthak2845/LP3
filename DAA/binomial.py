
n = int(input("Enter value of n: "))
k = int(input("Enter value of k: "))

# Create DP table
dp = [[0 for j in range(k + 1)] for i in range(n + 1)]

# Calculate binomial coefficient
for i in range(n + 1):
    for j in range(min(i, k) + 1):

        if j == 0 or j == i:
            dp[i][j] = 1
        else:
            dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]

print("Binomial Coefficient =", dp[n][k])

# Time Complexity: \(O(n \times k)\)
# Space Complexity: \(O(n \times k)\)