"""
题目：207. 课程表
链接：https://leetcode.cn/problems/course-schedule/description/

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
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        # 1. 建图 和 入度表
        adj = defaultdict(list)
        in_degree = [0] * numCourses
        
        for cur, pre in prerequisites:
            adj[pre].append(cur)
            in_degree[cur] += 1
            
        # 2. 将所有入度为 0 的节点放入队列
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        

        
        while queue:
            node = queue.popleft()
            numCourses -= 1
            
            # 遍历这门课的后续课程
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1 # 后续课程的入度减 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # 3. 如果修完的课程数等于总数，说明没有环
        return not numCourses
        
        
        
        
        
# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
    
    print(sol.canFinish(int(input()), list(input())))