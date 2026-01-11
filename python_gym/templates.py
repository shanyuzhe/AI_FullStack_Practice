"""
Python 刷题通用模板库 (Day 0 建立)
包含：快速I/O, 二分查找, BFS, 堆操作
"""
import sys
import heapq
from collections import deque
from typing import List

# ==========================================
# 1. 快速 I/O 模板 (解决大规模输入 TLE 问题)
# ==========================================
def fast_io():
    """
    用于 HackerRank / Codeforces 等需要自己处理输入的平台。
    LeetCode 通常不需要，但在处理大规模测试文件时有用。
    """
    input = sys.stdin.readline
    data = sys.stdin.read().split()
    return data

# ==========================================
# 2. 二分查找模板 (通用性最强版本)
# ==========================================
def binary_search(nums: List[int], target: int) -> int:
    """
    查找 target 在 nums 中的索引，找不到返回 -1
    核心思想：区间 [left, right] 闭区间
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search_left_bound(nums: List[int], target: int) -> int:
    """
    寻找左侧边界 (相当于 bisect_left)
    例如 [1, 2, 2, 2, 3], target=2 -> 返回索引 1
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

# ==========================================
# 3. BFS 模板 (层序遍历 / 最短路径)
# ==========================================
def bfs(start_node):
    """
    适用于：图的最短路径(无权)、树的层序遍历
    """
    if not start_node:
        return
    
    queue = deque([start_node])
    visited = {start_node} # 只有在图里才需要 visited，树不需要
    step = 0

    while queue:
        size = len(queue)
        # 遍历当前层的每一个节点
        for _ in range(size):
            curr = queue.popleft()
            
            # --- 业务逻辑处理 ---
            # if curr is target: return step
            
            # --- 扩展邻居节点 ---
            # for neighbor in curr.neighbors:
            #     if neighbor not in visited:
            #         visited.add(neighbor)
            #         queue.append(neighbor)
        step += 1

# ==========================================
# 4. 堆 (Heap) 常用操作
# ==========================================
def heap_usage_demo():
    """
    Python 的 heapq 默认是【小顶堆】(Min Heap)
    """
    # 1. 建堆 O(N)
    nums = [3, 1, 4, 1, 5, 9]
    heapq.heapify(nums) # nums 变为 [1, 1, 4, 3, 5, 9]
    
    # 2. 压入 O(logN)
    heapq.heappush(nums, 2)
    
    # 3. 弹出最小值 O(logN)
    min_val = heapq.heappop(nums)
    
    # 4. 如果需要【大顶堆】，技巧是存负数
    max_heap = [-x for x in [3, 1, 4]]
    heapq.heapify(max_heap)
    # 弹出最大值时，取负回来: -heapq.heappop(max_heap)