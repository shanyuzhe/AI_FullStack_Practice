"""
题目：609
链接：https://leetcode.cn/problems/find-duplicate-file-in-system/description/

复杂度分析：
- 时间复杂度：O(N * L)  
- 空间复杂度：O(N)

边界条件 (Edge Cases):
1. 
"""

import sys
from collections import Counter, defaultdict, deque
from typing import List
import heapq

class Solution:
    def solve(self, paths:List[str])-> List[List[str]]:
       # 变量名改为 groups，避免覆盖内置 dict
        groups = defaultdict(list)
        
        for info in paths:
            parts = info.split()
            root = parts[0]
            for f in parts[1:]:
                # 找左括号位置
                start = f.find('(')
                # 切片获取内容和文件名
                content = f[start + 1:-1]
                full_path = root + '/' + f[:start]
                
                groups[content].append(full_path)
        
        return [g for g in groups.values() if len(g) > 1]

# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    print(sol.solve(["root/a 1.txt(abcd) 2.txt(efgh)","root/c 3.txt(abcd)","root/c/d 4.txt(efgh)","root 4.txt(efgh)"]))
    
    