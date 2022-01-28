# Метод k-ближайших соседей.

from sklearn import datasets as dts
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
import timeit

# *Разбиваем датасет на две части: для теста и для данных проверки*

def split_dataset(data):
    len_mass = len(data.data) # длина массива данных
    mass_sort = [i for i in range(len_mass)] # массив индексов
    random.shuffle(mass_sort) # Рандомно сортируем индексы

    len_determine = int(len_mass * 0.7)
    # list_determine = [len_mass - len_determine, len_determine]
    datasets_learn = [[data.data[i], data.target[i]] for i in mass_sort[:len_determine]] # Первые 70 процентов добавляем для обучение,
    datasets_test = [[data.data[i], data.target[i]] for i in mass_sort[len_determine:]] # остальные для теста.
    return datasets_learn, datasets_test

# *Разбиваем датасет на две части: для теста и для данных проверки*
# *Нормализуем данные*

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
def l1(el1, el2):
    return sum([abs(k - v) for k, v in list(zip(el1, el2))])

def l2(el1, el2):
    return sum([(k - v)**2 for k, v in list(zip(el1, el2))]) ** 0.5

def l_infinity(el1, el2):
    max_len = 0
    for k, v in list(zip(el1, el2)):
        if abs(k- v) > max_len:
            max_len = abs(k - v)
    return max_len

# *Метод k-ближайших соседей*

def choice_class(mass_n):
    count = Counter(mass_n)
    return max(count, key = count.get)


def k_nearest_neightbors(data_learn, data_test, metric = l1, n = 4):
    rez = []
    for i in data_test:
        nearest_neighbors_list = []
        mass = []
        for j in data_learn:
            mass.append([metric(i[0], j[0]), j[1]])
        mass.sort(key = lambda a: a[0])
        for z in mass[:n]:
            nearest_neighbors_list.append(z[1])
        rez.append(choice_class(nearest_neighbors_list))
    return rez


def check_forecast(data_test, forecast):
    target_rez = len(data_test)
    forecast_rez = 0
    for i in range(len(data_test)):
        if data_test[i][1] == forecast[i]:
            forecast_rez += 1
    return forecast_rez / target_rez

def main():
    # Выполняем для "игрушечных" датасетов классификации.
    for i, name_dts in [(dts.load_iris, "Iris plants dataset"),
                        (dts.load_digits, "Optical recognition of handwritten digits dataset"),
                        (dts.load_wine,"Wine recognition dataset"),
                        (dts.load_breast_cancer, "Breast cancer wisconsin (diagnostic) dataset")]:
    # for i, name_dts in [(dts.load_iris, "Iris plants dataset"),
    #                 (dts.load_digits, "Optical recognition of handwritten digits dataset")]:


        data = i()

        # Разделение данных на тестовые и тренировочные.
        x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size = 0.7, random_state = 42)

        # Нормализуем данные.
        standartization(x_train)
        standartization(x_test)

        # Объединяем массивы данных с массивами целевых рез-тов.
        learn = np.array([[x, y] for x, y in zip(x_train, y_train)])
        test = np.array([[x, y] for x, y in zip(x_test, y_test)])

        # Выбираем кол-во соседей для обучения.
        # for j in [4]:
        for j in [4, 6, 8]:
            # Выбираем метрику.
            for m in [l1, l2, l_infinity]:
                rez = k_nearest_neightbors(learn, test, m, j)

                answer = check_forecast(test, rez)
                print(f"Обучение на данных {name_dts} с метрикой {m.__name__} для {j} ближайших соседей: {answer}")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    main()
    print(timeit.default_timer() - start_time)