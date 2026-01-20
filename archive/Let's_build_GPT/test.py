import itertools  

counter = itertools.count(start=1, step=2)
# def count(start=0, step=1):
#     # count(10) → 10 11 12 13 14 ...
#     # count(2.5, 0.5) → 2.5 3.0 3.5 ...
#     n = start
#     while True:
#         yield n
#         n += step
print(*(next(counter) for _ in range(5)))  # 1 3 5 7 9

cycle_counter = itertools.cycle(['A', 'B', 'C'])
print(*(next(cycle_counter) for _ in range(10)))  # A B C A B C A B C A

repeat_counter = itertools.repeat('Hello', 3)
print(*(next(repeat_counter) for _ in range(3)))  # Hello Hello Hello

product_counter = itertools.product([1, 2], ['A', 'B'])
print(*(next(product_counter) for _ in range(4)))  # (1, 'A') (1, 'B') (2, 'A') (2, 'B')

combination_counter = itertools.combinations(range(4), 2)
for i in combination_counter:
    print(i)

print("-----------------------------------------------------")
permutation_counter = itertools.permutations(range(4), 2)
for i in permutation_counter:
    print(i)
    

