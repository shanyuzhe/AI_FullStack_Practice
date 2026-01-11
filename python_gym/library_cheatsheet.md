# 🐍 Python 刷题常用库速查 (The Big 5)

## 1. collections (容器神器)
* **`Counter(iterable)`**: 统计词频。
    * `Counter(s).most_common(k)`:哪怕只为了这个 API 也要用它。
* **`defaultdict(type)`**: 自动处理键不存在的情况。
    * `d = defaultdict(list)`: 适合构建邻接表（图）。
    * `d = defaultdict(int)`: 适合计数（类似 Counter 但可动态加减）。
* **`deque`**: 双端队列。
    * `popleft()`, `appendleft()`: O(1) 时间复杂度（list 是 O(N)）。BFS 必备。

## 2. itertools (暴力枚举/迭代器)
* **`permutations(nums, k)`**: 排列 (有序)。A(n, k)
* **`combinations(nums, k)`**: 组合 (无序)。C(n, k)
* **`product(A, B)`**: 笛卡尔积，相当于嵌套循环。
* **`accumulate(nums)`**: 前缀和 (Prefix Sum)。

## 3. heapq (优先队列/堆)
* **默认是小顶堆**。
* `heapq.heappush(heap, item)`
* `heapq.heappop(heap)`
* `heapq.nlargest(k, nums)` / `nsmallest`: 取前 K 大/小元素（比先排序快）。

## 4. bisect (二分查找)
* **前提：数组必须有序**。
* `bisect.bisect_left(nums, x)`: 找插入位置，**>= x** 的第一个位置 (lower_bound)。
* `bisect.bisect_right(nums, x)`: 找插入位置，**> x** 的第一个位置 (upper_bound)。

## 5. functools (函数式技巧)
* **`@lru_cache(None)`**: **记忆化搜索神器**。加在 DFS 函数头上，自动缓存结果，DP 题变简单。
* `cmp_to_key`: 自定义比较函数用于 `sorted(key=...)`