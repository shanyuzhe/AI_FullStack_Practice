"""
题目：
链接：https://leetcode.cn/problems/ransom-note/

复杂度分析：
- 时间复杂度：O(N)
- 空间复杂度：O(1)

边界条件 (Edge Cases):
1. 
"""

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        from collections import Counter
        cnt_m =  Counter(magazine)
        for i in ransomNote:
            if cnt_m[i] <= 0:
                return False
            cnt_m[i] -= 1
        return True