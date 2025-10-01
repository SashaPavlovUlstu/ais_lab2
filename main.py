import numpy as np
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def triangular_mf(x, a, b, c):
    x = float(x)
    if a == b and b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0


class FuzzyLabel:
    def __init__(self, name, params):
        self.name = name
        self.a, self.b, self.c = params

    def mu(self, x):
        return triangular_mf(x, self.a, self.b, self.c)

    def mu_array(self, xs):
        vfunc = np.vectorize(lambda t: triangular_mf(t, self.a, self.b, self.c))
        return vfunc(xs)


class LinguisticVariable:
    def __init__(self, name, universe_min, universe_max, step=0.1):
        self.name = name
        self.labels = {}  # name -> FuzzyLabel
        self.universe = np.arange(universe_min, universe_max + 1e-9, step)

    def add_label(self, label_name, params):
        self.labels[label_name] = FuzzyLabel(label_name, params)

    def mu_vector_for_label(self, label_name):
        return self.labels[label_name].mu_array(self.universe)


def implication_mamdani_clip(muA_scalar, muB_vector):
    return np.minimum(muA_scalar, muB_vector)


def linguistic_match_degrees(mu_result_vector, consequent_variable: LinguisticVariable):
    degrees = {}
    for lname, label in consequent_variable.labels.items():
        mu_label = label.mu_array(consequent_variable.universe)
        intersect = np.minimum(mu_result_vector, mu_label)
        degrees[lname] = float(np.max(intersect))
    return degrees



def run_interactive():
    print('\n=== Модель импликации нечетких множеств (Mamdani, clipping=min) ===\n')
    print('Выберите режим:\n 1) Запустить пример\n 2) Ввести свои параметры')
    mode = input('Режим [1/2] (по умолчанию 1): ').strip() or '1'

    if mode == '1':
        # EXAMPLE: предметная область - Погода (Температура -> Комфорт)
        # antecedent variable "Температура" и одна метка A: "тепло"
        # consequent variable "Комфорт" с набором меток
        ant = LinguisticVariable('Температура (°C)', -10, 40, step=0.5)
        # метки для примера (треугольники): холодно, прохладно, тепло, жарко
        ant.add_label('холодно', (-10, -5, 5))
        ant.add_label('прохладно', (0, 7, 15))
        ant.add_label('тепло', (10, 20, 28))
        ant.add_label('жарко', (25, 32, 40))

        cons = LinguisticVariable('Уровень комфорта', 0, 10, step=0.1)
        cons.add_label('неудобно', (0.0, 0.0, 3.0))
        cons.add_label('терпимо', (2.0, 3.5, 5.0))
        cons.add_label('комфортно', (4.5, 6.0, 7.5))
        cons.add_label('очень комфортно', (7.0, 8.5, 10.0))

        print('\nПример: antecedent = "тепло" (треугольник (10,20,28)), consequent = метки комфорта\n')
        x_val = input('Введите четкое значение температуры в °C (по умолчанию 18): ').strip()
        x_val = float(x_val) if x_val else 18.0

        labelA = 'тепло'
        muA = ant.labels[labelA].mu(x_val)
        print(f'Степень принадлежности температуры {x_val}°C к метке "{labelA}" = {muA:.4f}')

        labelB = 'комфортно'
        muB_vec = cons.labels[labelB].mu_array(cons.universe)
        mu_result = implication_mamdani_clip(muA, muB_vec)

        degrees = linguistic_match_degrees(mu_result, cons)

        print('\nРезультат импликации (обрезанная метка "комфортно") представлен через метки переменной "Уровень комфорта":')
        for lname, d in sorted(degrees.items(), key=lambda x: -x[1]):
            print(f'  {lname:16s} : {d:.4f}')

        if HAS_MPL:
            plt.figure(figsize=(8,5))
            for lname, lab in cons.labels.items():
                plt.plot(cons.universe, lab.mu_array(cons.universe), label=f"{lname}")
            plt.fill_between(cons.universe, 0, mu_result, alpha=0.4, label='Результат импликации (mu_result)')
            plt.title(f'Импликация: если Температура = {x_val}°C то Комфорт')
            plt.xlabel(cons.name)
            plt.ylabel('mu')
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print('\nmatplotlib не доступен — график не будет показан.')

    else:
        print('\n--- Ввод параметров пользователем ---')
        # Ввод универсумов
        a_name = input('Имя antecedent переменной (по умолчанию Температура): ').strip() or 'Antecedent'
        a_min = float(input('Минимум универсама antecedent (по умолчанию 0): ').strip() or 0)
        a_max = float(input('Максимум универсама antecedent (по умолчанию 100): ').strip() or 100)
        a_step = float(input('Шаг универсама antecedent (по умолчанию 0.5): ').strip() or 0.5)
        ant = LinguisticVariable(a_name, a_min, a_max, step=a_step)

        n_a = int(input('Сколько меток для antecedent? (по умолчанию 1): ').strip() or 1)
        for i in range(n_a):
            lname = input(f'  Имя метки {i+1}: ').strip() or f'A{i+1}'
            a = float(input('    a: '))
            b = float(input('    b: '))
            c = float(input('    c: '))
            ant.add_label(lname, (a,b,c))

        x_val = float(input('Введите четкое значение x для antecedent: '))

        b_name = input('Имя consequent переменной (по умолчанию Consequent): ').strip() or 'Consequent'
        b_min = float(input('Минимум универсама consequent (по умолчанию 0): ').strip() or 0)
        b_max = float(input('Максимум универсама consequent (по умолчанию 10): ').strip() or 10)
        b_step = float(input('Шаг универсама consequent (по умолчанию 0.1): ').strip() or 0.1)
        cons = LinguisticVariable(b_name, b_min, b_max, step=b_step)

        n_b = int(input('Сколько меток для consequent? (по умолчанию 1): ').strip() or 1)
        for i in range(n_b):
            lname = input(f'  Имя метки {i+1}: ').strip() or f'B{i+1}'
            a = float(input('    a: '))
            b = float(input('    b: '))
            c = float(input('    c: '))
            cons.add_label(lname, (a,b,c))

        A_label = list(ant.labels.keys())[0]
        B_label = list(cons.labels.keys())[0]
        muA = ant.labels[A_label].mu(x_val)
        print(f'\nИспользуется antecedent метка "{A_label}" с muA={muA:.4f}')
        muB_vec = cons.labels[B_label].mu_array(cons.universe)
        mu_result = implication_mamdani_clip(muA, muB_vec)
        degrees = linguistic_match_degrees(mu_result, cons)
        print('\nРезультат импликации представлен через метки consequent:')
        for lname, d in sorted(degrees.items(), key=lambda x: -x[1]):
            print(f'  {lname:16s} : {d:.4f}')

        if HAS_MPL:
            plt.figure(figsize=(8,5))
            for lname, lab in cons.labels.items():
                plt.plot(cons.universe, lab.mu_array(cons.universe), label=f"{lname}")
            plt.fill_between(cons.universe, 0, mu_result, alpha=0.4, label='mu_result')
            plt.title('Результат импликации (пользовательский ввод)')
            plt.xlabel(cons.name)
            plt.ylabel('mu')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print('\nmatplotlib не доступен — график не будет показан.')


if __name__ == '__main__':
    run_interactive()
