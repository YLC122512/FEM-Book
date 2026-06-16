from main import *

if __name__ == "__main__":
    # 依次执行
    example_2_3_truss1()
    example_2_3_truss2()
    for n in [10, 100, 500]:
        example_tridiagonal(n)
    test_non_positive_definite()
    ill_conditioned_example()
    # 大规模 Poisson（根据机器性能调整）
    large_poisson(50, 50, solver='scipy')
    # 可选: large_poisson(100, 100)