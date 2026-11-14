N = 8

board = [[0 for _ in range(N)] for _ in range(N)]


def is_safe(row, col):

    for i in range(row):
        if board[i][col] == 1:
            return False
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True


def solve(row):

    if row == N:
        return True

    for col in range(N):

        if is_safe(row, col):
            board[row][col] = 1

            if solve(row + 1):
                return True

            board[row][col] = 0

    return False


r = int(input("Enter row of first queen (1-8): ")) - 1
c = int(input("Enter column of first queen (1-8): ")) - 1

if 0 <= r < N and 0 <= c < N:

    board[r][c] = 1

    if solve(r + 1):

        print("Solution:")
        for row in board:
            print(" ".join("Q" if x == 1 else "." for x in row))

    else:
        print("No solution exists for the given position.")

else:
    print("Invalid position!")
