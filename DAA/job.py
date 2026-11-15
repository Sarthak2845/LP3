def job_sequencing(jobs):
    jobs.sort(key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)

    slots = [None] * max_deadline

    total_profit = 0

    for job in jobs:
        job_id, deadline, profit = job

        for j in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[j] is None:
                slots[j] = job_id
                total_profit += profit
                break

    print("Scheduled Jobs:", [job for job in slots if job])
    print("Maximum Profit:", total_profit)


jobs = [
    ('J1', 2, 100),
    ('J2', 1, 19),
    ('J3', 2, 27),
    ('J4', 1, 25),
    ('J5', 3, 15)
]

job_sequencing(jobs)


# Final Answer (for exam):
# - TC: \(O(N\log N + N \times D)\)
# - SC: \(O(D)\)
# Steps of Greedy Algorithm:
# 1. Sort all jobs in decreasing order of profit.
# 2. Find the maximum deadline among all jobs.
# 3. Create empty time slots up to the maximum deadline.
# 4. Select jobs one by one according to their profit.
# 5. Place each job in the latest available slot before its deadline.
# 6. Calculate the total profit of all scheduled jobs.