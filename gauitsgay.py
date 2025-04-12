import random
from decimal import Decimal, getcontext
import numpy as np
from scipy.linalg import solve as scipy_solve
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if np.isclose(U[i, i], 0, atol=1e-10):
            x[i] = 0.0
        else:
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_with_lu(A, b):
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    L, U = lu_decomposition(A_np)
    y = forward_substitution(L, b_np)
    x = backward_substitution(U, y)
    return x


def gram_schmidt_qr(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i].astype(float)
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R


def solve_with_qr(A, b):
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    Q, R = gram_schmidt_qr(A_np)
    y = Q.T @ b_np
    x = scipy_solve(R, y)
    return x


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


def check_qr_for_rectangular():
    print("\n=== Проверка QR для не квадратной матрицы ===")
    m = int(input("Введите количество строк: "))
    n = int(input("Введите количество столбцов: "))

    A = []
    print(f"Введите матрицу {m}x{n} (по строкам, элементы через пробел):")
    for i in range(m):
        while True:
            try:
                row = list(map(lambda x: float(x.replace(',', '.')),
                               input(f"Строка {i + 1}: ").split()))
                if len(row) != n:
                    print(f"Ошибка: в строке должно быть {n} элементов.")
                    continue
                A.append(row)
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    try:
        A_np = np.array(A, dtype=float)
        Q, R = gram_schmidt_qr(A_np)

        np.set_printoptions(
            precision=8,
            suppress=True,
            floatmode="fixed"
        )

        print("\nМатрица Q:")
        print(Q)
        print("\nМатрица R:")
        print(R)

        QtQ = np.round(Q.T @ Q, 8)
        print("\nQ^T * Q:")
        print(QtQ)

        A_reconstructed = Q @ R
        print("\nВосстановленная матрица Q*R:")
        print(A_reconstructed)

    except Exception as e:
        print(f"Ошибка: {e}")

def check_lu_decomposition():
    print("\n=== Проверка LU-разложения ===")
    n = int(input("Введите размерность квадратной матрицы: "))

    A = []
    print(f"Введите матрицу {n}x{n} (по строкам, элементы через пробел):")
    for i in range(n):
        while True:
            try:
                row = list(map(float, input(f"Строка {i + 1}: ").split()))
                if len(row) != n:
                    print(f"Ошибка: в строке должно быть {n} элементов.")
                    continue
                A.append(row)
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    b = []
    print(f"Введите вектор b (элементы через пробел):")
    while True:
        try:
            b = list(map(float, input().split()))
            if len(b) != n:
                print(f"Ошибка: в векторе должно быть {n} элементов.")
                continue
            break
        except Exception as e:
            print(f"Ошибка: {e}")

    try:
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        L, U = lu_decomposition(A_np)
        print("\nматрица L:")
        for row in L:
            print("[", "  ".join(f"{x:10.6f}" for x in row), "]")
        print("\nматрица U:")
        for row in U:
            print("[", "  ".join(f"{x:10.6f}" for x in row), "]")
        det_L = np.prod(np.diag(L))
        det_U = np.prod(np.diag(U))
        det_A = det_L * det_U
        print("\nопределители:")
        print(f"det(L) = {det_L:.6f}")
        print(f"det(U) = {det_U:.6f}")
        print(f"det(A) = det(L) * det(U) = {det_A:.6f}")
        det_A_numpy = np.linalg.det(A_np)
        reconstructed = L @ U
        print("\nпроверка L*U:")
        for row in reconstructed:
            print("[", "  ".join(f"{x:10.6f}" for x in row), "]")

        y = forward_substitution(L, b_np)
        x = backward_substitution(U, y)

        print("\nрешение Ax = b:")
        for i, xi in enumerate(x, 1):
            print(f"x{i} = {xi:.6f}")

        residual = A_np @ x - b_np
        error = np.linalg.norm(residual)

    except Exception as e:
        print(f"Ошибка: {e}")
def main():
    getcontext().prec = 100

    while True:
        print("Выберите способ загрузки матрицы:")
        print("1. Клавиатура")
        print("2. Файл")
        print("3. Случайная генерация")
        print("4. Проверка QR-разложения")
        print("5. Проверка LU-разложения")
        try:
            method = input("Введите номер или название способа: ").lower()
        except EOFError:
            print("\nОбнаружен ввод Ctrl+D. Программа завершена.")
            return

        allowed_methods = ["1", "2", "3", "4", "5", "клавиатура", "файл", "случайная", "qr", "lu"]
        if method in allowed_methods:
            break
        else:
            print("Некорректный выбор. Пожалуйста, введите 1-4 или соответствующее название.")

    if method in ["4", "qr"]:
        check_qr_for_rectangular()
        return
    if method in ["5", "lu"]:
        check_lu_decomposition()
        return

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

    try:
        matrix_norm = operator_norm(A)
        print(f"\nОператорная норма матрицы: {matrix_norm}")
    except Exception as e:
        print(f"\nОшибка при вычислении операторной нормы: {e}")

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

    x_lu = solve_with_lu(A, b)
    print("Решение методом LU-разложения:")
    for i, sol in enumerate(x_lu, start=1):
        print(f"x{i} = {sol}")

    x_qr = solve_with_qr(A, b)
    print("Решение методом QR-разложения:")
    for i, sol in enumerate(x_qr, start=1):
        print(f"x{i} = {sol}")

def generate_random_matrix(n):
    A = [[Decimal(0)] * n for _ in range(n)]
    for i in range(n):
        diagonal_value = Decimal(random.uniform(100, 200))
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
