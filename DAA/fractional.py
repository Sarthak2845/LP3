
n = int(input("Enter number of items: "))

profit = []
weight = []

print("Enter profit of each item:")
for i in range(n):
    profit.append(float(input()))

print("Enter weight of each item:")
for i in range(n):
    weight.append(float(input()))

capacity = float(input("Enter capacity of knapsack: "))

# Calculate profit/weight ratio
ratio = []

for i in range(n):
    ratio.append((profit[i] / weight[i], i))

# Sort by ratio in descending order
ratio.sort(reverse=True)

total_profit = 0
selected = []

for r, i in ratio:
    if capacity == 0:
        break

    if weight[i] <= capacity:
        fraction = 1
        total_profit += profit[i]
        capacity -= weight[i]
    else:
        fraction = capacity / weight[i]
        total_profit += profit[i] * fraction
        capacity = 0

    selected.append((i + 1, fraction))

print("Maximum Profit =", total_profit)
print("Selected items:")

for item, fraction in selected:
    print(f"Item {item} - Fraction: {fraction:.2f}".rstrip('0').rstrip('.'))
