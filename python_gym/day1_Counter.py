"""
题目：Counter API
链接：https://www.hackerrank.com/challenges/collections-counter/problem?isFullScreen=true

复杂度分析：
- 时间复杂度：O(N)
- 空间复杂度：ON)

边界条件 (Edge Cases):
1. 
"""
import sys
from collections import Counter, defaultdict, deque
from typing import List
import heapq

class Solution:
    def solve(self, shop_shoes, customers) -> None:
        shoe_count = Counter(shop_shoes)
        sum = 0
        for size, price in customers:
            if shoe_count[size] > 0:
                sum += price
                shoe_count[size] -= 1
        return sum

# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    x = int(input())
    shop_shoes = list(map(int, input().split()))
    n = int(input())
    customers = [tuple(map(int,input().split())) for _ in range(n)]
    print(sol.solve(shop_shoes, customers))