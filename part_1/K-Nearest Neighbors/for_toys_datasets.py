
from sklearn import datasets as dts
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
import timeit
import numba

# *Разбиваем датасет на две части: для теста и для данных проверки*
@numba.njit
def split_dataset(data):
    len_mass = len(data.data) # длина массива данных
    mass_sort = [i for i in range(len_mass)] # массив индексов
    random.shuffle(mass_sort) # рандомно сортируем индексы

    len_determine = int(len_mass * 0.7)
    # list_determine = [len_mass - len_determine, len_determine
    datasets_learn = [[data.data[i], data.target[i]] for i in mass_sort[:len_determine]] # Первые 70 процентов добавляем для обучение,
    datasets_test = [[data.data[i], data.target[i]] for i in mass_sort[len_determine:]] # остальные для теста.
    return datasets_learn, datasets_test

# *Разбиваем датасет на две части: для теста и для данных проверки*
# *Нормализуем данные*

@numba.njit
def standartization(data):
    # Находим макс и мин эл.
    min_elem = np.min(data)
    max_elem = np.max(data)

    difference = max_elem - min_elem

    # Меняем значения на нормализованные.
    for i in range(len(data)):
        data[i] = (data[i] - min_elem) / difference

# *Метрики*
# Вход el1 = [el1, el2, ... , eln]
@numba.njit
def l1(el1, el2):
    return sum([abs(k - v) for k, v in list(zip(el1, el2))])

@numba.njit
def l2(el1, el2):
    return sum([(k - v)**2 for k, v in list(zip(el1, el2))]) ** 0.5

@numba.njit
def l_infinity(el1, el2):
    return max([abs(k - v) for k, v in list(zip(el1, el2))])

def choice_class(mass_n):
    count = Counter(mass_n)
    return max(count, key = count.get)

def k_nearest_neightbors(x_train, x_test, y_train, metric = l1, n = 4):
    neightbors = np.array([np.array([metric(x, y) for y in x_train]).argsort()[:n] for x in x_test])
    rez = [choice_class([y_train[j] for j in i]) for i in neightbors]
    return rez

def check_forecast(y_test, forecast):
    target_rez = len(y_test)
    forecast_rez = 0
    for i in range(len(y_test)):
        if y_test[i] == forecast[i]:
            forecast_rez += 1

    return forecast_rez / target_rez

def main():
    for i, name_dts in [(dts.load_iris, "Iris plants dataset"),
                    (dts.load_digits, "Optical recognition of handwritten digits dataset"),
                    (dts.load_wine,"Wine recognition dataset"),
                    (dts.load_breast_cancer, "Breast cancer wisconsin (diagnostic) dataset")]:

        data = i()

        x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size = 0.7, random_state = 42)

        standartization(x_train)
        standartization(x_test)
        greatest_rez = [0, 0, 0]

        for n in range(4, 10):
            for m in [l1, l2, l_infinity]:

                rezult = k_nearest_neightbors(x_train, x_test, y_train, m, n)

                answer = check_forecast(y_test, rezult)
                if greatest_rez[0] < answer:
                    greatest_rez[0] = answer
                    greatest_rez = [answer, m, n]
                # logreg.predict(X_test) # Прогнозируемые данные
        print(f"Обучение на данных {name_dts} с метрикой {greatest_rez[1].__name__} для {greatest_rez[2]} ближайших соседей: {greatest_rez[0]}")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    main()
    print(timeit.default_timer() - start_time)