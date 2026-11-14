n = int(input("Enter number of items: "))

profit = []
weight = []

print("Enter profit of each item:")
for i in range(n):
    profit.append(int(input()))

print("Enter weight of each item:")
for i in range(n):
    weight.append(int(input()))

W = int(input("Enter capacity of knapsack: "))

dp = [[0 for j in range(W + 1)] for i in range(n + 1)]

for i in range(1, n + 1):
    for j in range(W + 1):

        if weight[i - 1] <= j:
            dp[i][j] = max(
                profit[i - 1] + dp[i - 1][j - weight[i - 1]],
                dp[i - 1][j]
            )
        else:
            dp[i][j] = dp[i - 1][j]

print("Maximum Profit =", dp[n][W])

selected = []
j = W

for i in range(n, 0, -1):
    if dp[i][j] != dp[i - 1][j]:
        selected.append(i)
        j -= weight[i - 1]

print("Selected items:")
for item in reversed(selected):
    print("Item", item)
