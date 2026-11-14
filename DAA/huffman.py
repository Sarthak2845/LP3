
import heapq

def huffman(s):
    freq = {}

    # Calculate frequency
    for ch in s:
        if ch in freq:
            freq[ch] += 1
        else:
            freq[ch] = 1

    pq = []

    # Create nodes
    for ch in freq:
        heapq.heappush(pq, [freq[ch], ch, None, None])

    # Build Huffman Tree
    while len(pq) > 1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)

        new_node = [
            left[0] + right[0],
            left[1] + right[1],
            left,
            right
        ]

        heapq.heappush(pq, new_node)

    codes = {}

    # Generate codes
    def generate(root, code):
        if root[2] is None and root[3] is None:
            codes[root[1]] = code
            return

        generate(root[2], code + "0")
        generate(root[3], code + "1")

    generate(pq[0], "")

    print("\nString:", s)
    print("\nHuffman Code Table:")
    print("Character\tFrequency\tCode")

    for ch in sorted(freq):
        print(ch, "\t\t", freq[ch], "\t\t", codes[ch])

    encoded = ""

    for ch in s:
        encoded += codes[ch]

    print("\nEncoded String:", encoded)


s = input("Enter string: ")
huffman(s)

# Time: \(O(n\log n)\)
# Space: \(O(n)\)