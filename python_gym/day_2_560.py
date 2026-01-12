"""
题目：560. 和为 K 的子数组
链接：https://leetcode.cn/problems/subarray-sum-equals-k/description/

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
    def subarraySum(self, nums: List[int], k: int) -> int:
       
        #mp = {0:1}
        mp = defaultdict(int)
        mp[0] = 1
        cnt = 0
        pre = 0 #前缀和
        for num in nums:
            pre += num
            if (pre - k) in mp:
                cnt += mp[pre - k]
            
            mp[pre] += 1
            #普通字典的写法
            #mp[pre] = mp.get(pre, 0) + 1
            #value = 字典.get(key, default_value)
            #找不到key的话返回default_value 避免访问空位置报错

        return cnt

# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    print(sol.subarraySum(nums = [1,2,3],k = 1))