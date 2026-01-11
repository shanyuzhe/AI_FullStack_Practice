"""
题目：346
链接：

复杂度分析：
- 时间复杂度：O(N)
- 空间复杂度：O(1)

边界条件 (Edge Cases):
1. 
"""

import sys
from collections import Counter, defaultdict, deque
from typing import List
import heapq

class Solution:
    def solve(self, nums, k):
        cnt = Counter(nums)
        ans = []
        for num in cnt:
            print(num)
            ans.append(num)
        return ans
    #most_common解释：返回一个列表，列表中的元素是按频率从高到低排序的元组 (元素, 频率)
    #遍历most_common(k)返回前k个高频元素

# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    print(sol.solve([3,1,1,2,4], 2))