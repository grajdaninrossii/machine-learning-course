
import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn import datasets as dts
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random

# Находим gini на выборке
def gini_impurity(y):
    try:
        _, counts = np.unique(y, return_counts = True)
        p = counts / y.shape[0]
        gini = 1 - np.sum(p**2)
        return(gini)
    except:
       raise TypeError("Ошибка, в функцию gini_impurity() нужно вводить массив!")

# Находим энтропию на выборке
def entropy(y):
    try:
        _, counts = np.unique(y, return_counts = True)
        a = counts / y.shape[0]
        entropy = np.sum(-a * np.log2(a))
        return(entropy)
    except:
        raise TypeError("Ошибка, в функцию entropy() нужно вводить массив!")

# Xm - выборка до разбиения
# Xl, Xr - выборки полученные разбиением (целевые значения, т.е. классы)
def gain_info(criterion, Xm, Xl, Xr):
    q = criterion(Xm) - Xl.shape[0] / Xm.shape[0] * criterion(Xl) - Xr.shape[0] / Xm.shape[0] * criterion(Xr)
    return q, criterion(Xm)

# Находим наилучшее условие для разбиения
def create_split(X, y, criterion, signs):

    # Подсчитываем кол-во каждого класса
    values = tuple([y[y == v].shape[0] for v in signs])
    # print(values)
    term = [(0, 0), 0, 0, y.shape[0], values, max(enumerate(values), key=lambda x: x[1])[0]]

    # [условие, criterion, gain_info, samples, values, class]
    if not criterion(y):
        term[0] = False
        return term

    # Смотрим по признаку
    for i in range(X.shape[1]):
        u_x_i = np.unique(X[:, i])

        # Выбираем пороговое значение
        for j in range(u_x_i.shape[0] - 1):
            cnd = (u_x_i[j] + u_x_i[j + 1]) / 2
            new_term = [(i, cnd), *gain_info(criterion, y, y[X[:, i] >= cnd],  y[X[:, i] < cnd])]
            if term[1] <= new_term[1]:
                term[:3] = new_term
    return term

# Строим дерево решений
# Разбиение данасета по условию
def split_dt(X, y, splt):
    if not splt[0]:
        return -1 # Ошибка
    c = splt[0]
    return [
                X[X[:, c[0]] < c[1]],
                y[X[:, c[0]] < c[1]],
                X[X[:, c[0]] >=  c[1]],
                y[X[:, c[0]] >= c[1]]
            ]

# Ячейка дерева, где
# X, y - data, target, фильтрованные по условию родителя
# signs - уникальные значения y, до разбиения
# max_depth - максимальная глубина дерева
# depth - начальный уровень дерева
def node(X, y, criterion, signs, max_depth, depth = 1):
    split = [*create_split(X, y, criterion, signs), 1]

    if not split[0] or depth > max_depth - 1:
        # print("Это лист", split[1:])
        split[-1] = 0
        return split
    Xl, yl, Xr, yr = split_dt(X, y, split)

    # Рекурсией создаем дерево
    return split, node(Xl, yl, criterion, signs, max_depth, depth + 1), node(Xr, yr, criterion, signs, max_depth, depth + 1)

import math
# Создаем дерево, вызывая функцию для начала рекурсии
def tree(X, y, criterion = gini_impurity, max_depth = math.inf):
    clf = node(X, y, criterion, np.unique(y), max_depth)
    return clf

def predict_proba(node, X):
    # print(node[0], "-----\n", node[2])
    # if not node[0][-1]:
    #     return node[0][-2]
    cond_index = node[0][0][0]
    cond_value = node[0][0][1]
    if X[cond_index] <= cond_value:
        if node[1][-1]:
            # print("Левая ветка")
            return predict_proba(node[1], X)
        else:
            # print("Наш ответ", node[1][-2])
            return node[1][-2]
    else:
        if node[2][-1]:
            # print("Правая ветка")
            return predict_proba(node[2], X)
        else:
            # print("Наш ответ", node[2][-2])
            return node[2][-2]

def score(tree, X, y):
    rez = np.array(predict(tree, X))
    return rez[y == rez].size / rez.size

# Классификатор массива
def predict(tree, X):
    return [predict_proba(tree, x) for x in X]

# Просмотреть дерево в форме массивов
def view_tree_mass(node, tab = ""):
    print(tab, node[0])
    if node[0][-1]:
        if node[1][-1]:
            view_tree_mass(node[1], tab + "\t")
        else:
            print(tab + "\t", node[1])
        if node[2][-1]:
            view_tree_mass(node[2], tab + "\t")
        else:
            print(tab + "\t", node[2])

# Формализуем значения ячейки
def create_label(lbl, feature_names, class_names):
    new_lbl = ""
    if lbl[0]:
        feature = feature_names[lbl[0][0]]
        feature_value = lbl[0][1]
        new_lbl = f"{feature} <= {feature_value}\n"
    gini = lbl[2]
    samples = lbl[3]
    value = lbl[4]
    cls = class_names[lbl[5]]
    new_lbl += f"gini = {gini:.3f}\n" + \
        f"samples = {samples}\n" +\
        f"value = {value}\n" +\
        f"class = {cls}"
    return new_lbl

# Используем формат hsla
def create_color(cls):
    visibility = "ff"

    # Задаем готовые цвета, ибо в алгоритм генерации я быстро не смог...
    # Если классов будет больше, чем цветов в списке, то вернет белый цвет
    colors = ["#ffffff", "#0000ff", "#00ff00", "#00ffff", "#ff0000", "#ff00ff", "#ffff00", "#00ffaa", "#00aaff", "#aa0000", "#ff00aa", "#aa00ff"]
    # l = list(range(0, 10, 2))
    # random.shuffle(l)
    # l = [v * 0.1 for v in l]

    # Смотрим, есть ли явный кандитат или нет
    set_cls = [c for c in cls if c]
    # print(set_cls)

    if len(set_cls) != len(set(set_cls)):
        return colors[0] + visibility
        # print("хз")
    elif len(set(set_cls)) > 1:
        visibility = "20"
    predict_i = max(enumerate(cls), key=lambda x: x[1])[0]
    try:
        return colors[predict_i + 1] + visibility
    except:
        return colors[0] + visibility

        # c = colors[predict_i % len(colors)].split(" ")
        # "".join(c) if predict_i > len(colors) else "".join(c[0] - 0.3 > 0 if c[0] - 0.3 else c[0], 3)
        # k =  c[0] - 0.3 > 0 if c[0] - 0.3 else c[0]


# Просмотреть дерево в виде графа
def view_tree_graph(node, dot, k, feature_names, class_names):
    dot.node(f"{k + 1}" +  f"{node[0]}", label = create_label(node[0], feature_names, class_names), style='filled', fillcolor=create_color(node[0][-3]))
    if node[0][-1]:
        if node[1][-1]:
            view_tree_graph(node[1], dot, k + 1, feature_names, class_names)
            dot.edge(f"{k + 1}" +  f"{node[0]}", f"{k + 2}" +  f"{node[1][0]}")
        else:
            dot.node(f"{k + 1}" +  f"{node[1]}", label = create_label(node[1], feature_names, class_names), style='filled', fillcolor=create_color(node[1][-3]))
            dot.edge(f"{k + 1}" +  f"{node[0]}", f"{k + 1}" +  f"{node[1]}")
        if node[2][-1]:
            view_tree_graph(node[2], dot, k + 1, feature_names, class_names)
            dot.edge(f"{k + 1}" +  f"{node[0]}", f"{k + 2}" +  f"{node[2][0]}")
        else:
            dot.node(f"{k + 1}" +  f"{node[2]}", label = create_label(node[2], feature_names, class_names), style='filled', fillcolor=create_color(node[2][-3]))
            dot.edge(f"{k + 1}" +  f"{node[0]}", f"{k + 1}" +  f"{node[2]}")

# Просмотреть дерево в виде графа
def make_graph(tree, feature_names, class_names):
    dot = Digraph(comment='The Round Table', )
    dot.attr("node", shape = "rectangle")
    view_tree_graph(tree, dot, 10, feature_names, class_names)
    return dot

def main():
    for load_dts, name_dts in [(dts.load_iris, "Iris plants dataset"),
                               (dts.load_digits, "Optical recognition of handwritten digits dataset"),
                               (dts.load_wine,"Wine recognition dataset"),
                               (dts.load_breast_cancer, "Breast cancer wisconsin (diagnostic) dataset")]:

        print(name_dts, ":", sep="")

        data = load_dts(); # Загружаем датасеты

        # Разделяем данные
        X = data.data
        y = data.target
        # print(np.unique(y))

        # Делим данные на тестовые и тренировочные
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42)

        # Создаем дерево (все в массивах), tree_train - итоговый массив
        tree_train = tree(x_train, y_train, criterion = entropy)

        # view_tree_mass(tree_train)

        # Выводим на графике
        graphic = make_graph(tree_train, data.feature_names, data.target_names)
        graphic.render(f"graphics/{name_dts}.gv", view=True)

        print(score(tree_train, x_test, y_test))
        # print(y_test)

        print("---")
        # print(tree(X, y))

if __name__ == "__main__":
    main()