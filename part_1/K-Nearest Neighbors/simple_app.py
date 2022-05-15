# Метод k-ближайших соседей.

from sklearn import datasets as dts
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import timeit


def main():
    # Загружаем датасеты по очереди.
    for load_dataset, name_dts in [(dts.load_iris, "Iris plants dataset"),
                        (dts.load_digits, "Optical recognition of handwritten digits dataset"),
                        (dts.load_wine,"Wine recognition dataset"),
                        (dts.load_breast_cancer, "Breast cancer wisconsin (diagnostic) dataset")]:

        data = load_dataset()

        X = data.data
        y = data.target
        # разделим данные с помощью Scikit-Learn's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42)

        # Выполним нормализвацию
        #scaler = preprocessing.StandardScaler() # Создаем объект для нормализации

        min_max_scaler = preprocessing.MinMaxScaler() # Создаем объект для нормализации
        # min_max_scaler.fit(X_train) # Подготавливает данные к маштабированию
        # min_max_scaler.fit(X_test)

        # min_max_scaler.transform(X_train) # Маштабирование
        # min_max_scaler.transform(X_test)
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)


        metrics_list = ["euclidean", "manhattan", "chebyshev"]
        greatest_rez = [0, 0, 0]
        for n in range(4, 10):
            for m in metrics_list:
                logreg = KNeighborsClassifier(metric = m, n_neighbors = n) # Тренируем
                logreg.fit(X_train, y_train)
                answer = logreg.score(X_test, y_test) # Сравнение прогнозов с правильными рез-тами.

                if greatest_rez[0] < answer:
                    greatest_rez[0] = answer
                    greatest_rez = [answer, m, n]
                # logreg.predict(X_test) # Прогнозируемые данные
        print(f"Обучение на данных {name_dts} с метрикой {greatest_rez[1]} для {greatest_rez[2]} ближайших соседей: {greatest_rez[0]}")

if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    print(timeit.default_timer() - start)