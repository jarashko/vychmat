import random
from decimal import Decimal, getcontext
import numpy as np
from scipy.linalg import lu, solve as scipy_solve
from numpy.linalg import qr as numpy_qr, norm as numpy_norm

max_iter = 100000

def dot_product(a, b):
    result = Decimal(0)
    for i in range(len(a)):
        result += Decimal(a[i]) * Decimal(b[i])
    return result

def norm(vector):
    result = Decimal(0)
    for val in vector:
        result += Decimal(val) ** Decimal(2)
    return result ** Decimal(0.5)

getcontext().prec = 100

def matrix_vector_mult(A, x):
    return [dot_product(row, x) for row in A]

def transpose(A):
    return list(map(list, zip(*A)))

def operator_norm(A, max_iterations=1000, tolerance=Decimal('1e-10')):
    n = len(A)
    if n == 0:
        return Decimal(0)

    x = [Decimal(str(random.random())) for _ in range(n)]
    x_norm = norm(x)
    if x_norm == Decimal(0):
        return Decimal(0)
    x = [xi / x_norm for xi in x]

    A_t = transpose(A)

    for _ in range(max_iterations):
        Ax = matrix_vector_mult(A, x)
        AtAx = matrix_vector_mult(A_t, Ax)

        new_norm = norm(AtAx)
        if new_norm == Decimal(0):
            return Decimal(0)
        x_new = [xi / new_norm for xi in AtAx]

        delta = norm([x_new[i] - x[i] for i in range(n)])
        x = x_new
        if delta < tolerance:
            break

    Ax = matrix_vector_mult(A, x)
    AtAx = matrix_vector_mult(A_t, Ax)
    lambda_max = dot_product(AtAx, x)

    return lambda_max.sqrt()

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = [Decimal(val) for val in x0]
    iterations = 0
    residual_norm = Decimal('Infinity')

    while residual_norm > tol and iterations < max_iter:
        x_new = x.copy()
        for i in range(n):
            sigma = dot_product(A[i][:i], x_new[:i]) + dot_product(A[i][i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sigma) / A[i][i]

        residual = [dot_product(A[i], x_new) - b[i] for i in range(n)]
        residual_norm = norm(residual)

        delta_norm = norm([x_new[i] - x[i] for i in range(n)])
        x = x_new

        iterations += 1
        print(f"Итерация {iterations}: Норма невязки = {residual_norm}, Изменение решения = {delta_norm}")

        if delta_norm < Decimal('1E-' + str(getcontext().prec // 2)):
            print("Изменение решения стало слишком малым. Завершение итераций.")
            break

    return x, iterations

def check_diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        diagonal_value = abs(Decimal(A[i][i]))
        sum_other_elements = sum(abs(Decimal(A[i][j])) for j in range(n) if j != i)
        if diagonal_value <= sum_other_elements:
            return False
    return True

def rearrange_for_diagonal_dominance(A, b):
    n = len(b)
    row_order = sorted(range(n), key=lambda i: -abs(A[i][i]))
    A_reordered = [A[i] for i in row_order]
    b_reordered = [b[i] for i in row_order]

    if check_diagonal_dominance(A_reordered):
        return A_reordered, b_reordered

    A_transposed = list(map(list, zip(*A)))
    col_order = sorted(range(n), key=lambda i: -abs(A_transposed[i][i]))
    A_transposed_reordered = [A_transposed[i] for i in col_order]
    A_reordered = list(map(list, zip(*A_transposed_reordered)))

    if check_diagonal_dominance(A_reordered):
        return A_reordered, b

    return None, None

def load_matrix_from_file(filename):
    try:
        with open(filename, 'r') as file:
            n = int(file.readline())
            tol = Decimal(file.readline().replace(',', '.'))
            if tol <= Decimal(0):
                print("Ошибка: Точность должна быть положительным числом.")
                return None, None, None
            A = []
            b = []
            for _ in range(n):
                row = list(map(lambda x: Decimal(x.replace(',', '.')),
                               file.readline().strip().split()))
                A.append(row[:-1])
                b.append(row[-1])
            return A, b, tol
    except FileNotFoundError:
        print("Ошибка: Файл не найден.")
        return None, None, None
    except PermissionError:
        print("Ошибка: Нет доступа к файлу. Проверьте права доступа.")
        return None, None, None
    except ValueError as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, None, None

def solve_with_lu(A, b):
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    P, L, U = lu(A_np)
    y = scipy_solve(L, b_np)
    x = scipy_solve(U, y)
    return x

def solve_with_qr(A, b):
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    Q, R = numpy_qr(A_np)
    y = Q.T @ b_np
    x = scipy_solve(R, y)
    return x

def main():
    getcontext().prec = 100

    while True:
        print("Выберите способ загрузки матрицы:")
        print("1. Клавиатура")
        print("2. Файл")
        print("3. Случайная генерация")
        try:
            method = input("Введите номер или название способа: ").lower()
        except EOFError:
            print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
            return

        if method in ["1", "клавиатура", "2", "файл", "3", "случайная"]:
            break
        else:
            print("Некорректный выбор. Пожалуйста, введите 1, 2, 3 или соответствующее название.")

    if method == "1" or method == "клавиатура":
        while True:
            try:
                n = int(input("Введите размерность матрицы: "))
                if n > 20:
                    print("Ошибка: Размерность матрицы должна быть не более 20.")
                    continue
                break
            except ValueError:
                print("Ошибка: Введите корректное целое число для размерности матрицы.")
            except EOFError:
                print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                return

        A = [[Decimal(0)] * n for _ in range(n)]
        b = [Decimal(0)] * n
        for i in range(n):
            while True:
                try:
                    print(f"Введите коэффициенты {i + 1} строки через пробел (всего {n + 1} чисел: {n} коэффициентов матрицы и 1 элемент вектора b):")
                    row = list(map(lambda x: Decimal(x.replace(',', '.')),
                                   input().split()))
                    if len(row) != n + 1:
                        print(f"Ошибка: В строке должно быть {n + 1} чисел.")
                        continue
                    A[i] = row[:-1]
                    b[i] = row[-1]
                    break
                except ValueError:
                    print(f"Ошибка: Некорректный ввод коэффициентов в строке {i + 1}. Проверьте введенные данные.")
                except EOFError:
                    print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                    return

        while True:
            try:
                tol = Decimal(input("Введите точность: ").replace(',', '.'))
                if tol <= Decimal(0):
                    print("Ошибка: Точность должна быть положительным числом.")
                    continue
                break
            except ValueError:
                print("Ошибка: Некорректный ввод точности. Проверьте введенные данные.")
            except EOFError:
                print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                return

    elif method == "2" or method == "файл":
        while True:
            try:
                filename = input("Введите имя файла с матрицей: ")
                A, b, tol = load_matrix_from_file(filename)
                if A is not None and b is not None and tol is not None:
                    break
                else:
                    print("Попробуйте ввести имя файла снова или выберите другой способ загрузки.")
                    choice = input("Хотите попробовать снова? (да/нет): ").lower()
                    if choice != "да":
                        return
            except EOFError:
                print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                return

        n = len(A)
        if n > 20:
            print("Ошибка: Размерность матрицы должна быть не более 20.")
            return

    elif method == "3" or method == "случайная":
        while True:
            try:
                n = int(input("Введите размерность матрицы: "))
                if n > 20:
                    print("Ошибка: Размерность матрицы должна быть не более 20.")
                    continue
                break
            except ValueError:
                print("Ошибка: Введите корректное целое число для размерности матрицы.")
            except EOFError:
                print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                return

        A = generate_random_matrix(n)
        b = [Decimal(random.uniform(-100, 100)) for _ in range(n)]

        while True:
            try:
                tol = Decimal(input("Введите точность: ").replace(',', '.'))
                if tol <= Decimal(0):
                    print("Ошибка: Точность должна быть положительным числом.")
                    continue
                break
            except ValueError:
                print("Ошибка: Некорректный ввод точности. Проверьте введенные данные.")
            except EOFError:
                print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
                return

        print("Сгенерированная матрица A:")
        print_matrix(A)
        print("Сгенерированный вектор b:")
        print(b)

    while True:
        try:
            max_iter = int(input("Введите максимальное количество итераций: "))
            if max_iter <= 0:
                print("Ошибка: Количество итераций должно быть положительным числом.")
                continue
            break
        except ValueError:
            print("Ошибка: Некорректный ввод количества итераций. Проверьте введенные данные.")
        except EOFError:
            print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
            return

    if not check_diagonal_dominance(A):
        print("В исходной матрице отсутствует диагональное преобладание. Переставляем строки/столбцы.")
        A, b = rearrange_for_diagonal_dominance(A, b)
        if A is None or b is None:
            print("Ошибка: Невозможно достичь диагонального преобладания.")
            return

    matrix_norm = operator_norm(A)
    print(f"Операторная Норма матрицы: {matrix_norm}")

    x0 = [Decimal(1)] * n

    x, iterations = gauss_seidel(A, b, x0, tol, max_iter)

    print("Решение методом Гаусса-Зейделя:")
    for i, sol in enumerate(x, start=1):
        print(f"x{i} = {sol}")

    print(f"Количество итераций: {iterations}")

    residual = [sum(Decimal(A[i][j]) * Decimal(x[j]) for j in range(n)) - Decimal(b[i]) for i in range(n)]
    print("Вектор погрешностей:")
    for i, res in enumerate(residual, start=1):
        print(f"r{i} = {res}")

    residual_norm = norm(residual)
    print(f"Норма невязки: {residual_norm}")

    if residual_norm < tol:
        print("Требуемая точность достигнута.")
    else:
        print("Требуемая точность не достигнута.")

    # Решение с использованием LU-разложения
    x_lu = solve_with_lu(A, b)
    print("Решение методом LU-разложения:")
    for i, sol in enumerate(x_lu, start=1):
        print(f"x{i} = {sol}")

    # Решение с использованием QR-разложения
    x_qr = solve_with_qr(A, b)
    print("Решение методом QR-разложения:")
    for i, sol in enumerate(x_qr, start=1):
        print(f"x{i} = {sol}")

def generate_random_matrix(n):
    A = [[Decimal(0)] * n for _ in range(n)]
    for i in range(n):
        diagonal_value = Decimal(random.uniform(100, 200))  # Большое значение
        A[i][i] = diagonal_value

        for j in range(n):
            if j != i:
                A[i][j] = Decimal(random.uniform(-10, 10))

        sum_other_elements = sum(abs(A[i][j]) for j in range(n) if j != i)
        if A[i][i] <= sum_other_elements:
            A[i][i] = sum_other_elements + Decimal(random.uniform(1, 10))

    return A

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()
