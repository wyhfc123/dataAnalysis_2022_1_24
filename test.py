# with open("sklearn_data/lenses.data") as f:
#     def strip_space(l):
#         print(l)
#
#     lenses = [inst.strip().split() for inst in f.readlines()]
#     print(lenses)


# print(list(map(int,["a","2"])))
# import numpy as np
# with open("sklearn_data/lenses.data") as f:
#     lenses = [inst.strip("\n").split("  ") for inst in f.readlines()]
#     lenses = np.array(lenses)
#     result_set = lenses[:,-1]
#     print(result_set[0])
#     result_set_label = ["hard contact lenses", "soft contact lenses", "no contact lenses"]
#
#
#
#     # print(np.where(map(result_set,result_set_label), result_set, result_set_label))
#     lensesLabels = ["age","prescript","astigmatic","tearRate"]
# print(np.power(3,2))


# #numpy 和 sympy合用
# import numpy as np
# import sympy as sp
# xm = np.array([-np.pi/2,0,np.pi/2])
#
# x= sp.symbols("x")
# z =sp.sin(x)
# f = sp.lambdify(("x"),z,"numpy")
# print(f(xm))