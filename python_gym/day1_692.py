"""
题目：692
链接：https://leetcode.cn/problems/top-k-frequent-words/

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
    def solve(self, words: List[str], k: int) -> List[str]:
        cnt = Counter(words)
        items = sorted(cnt.items(), key = lambda x:(-x[1],x[0]))
        return [w for w,_ in items[:k]]
        #这个lambda表达式的意思是，先按照频率从大到小排序（-x[1]），如果频率相同，则按照字典序从小到大排序（x[0]）
        #注意不能写反 对字符串取负号 会报错
# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    print(sol.solve(["i","love","leetcode","i","love","coding"], 2))