import numpy as np

# def k():
#     while True:
#         yield 1
#
# for epoch,data in enumerate(k):
#     print(epoch,data)

a = 1
b = None
bb = 2
c = bb or a
d = a or bb
print(c,d)