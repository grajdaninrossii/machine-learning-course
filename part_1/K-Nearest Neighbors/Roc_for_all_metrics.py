
# # ROC - кривая
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import timeit

def creat_graph(fpr, tpr, roc_auc):

    # строим график
    for i in range(6):
        color_graph = ["black", "red", "orange", "yellow", "lime", "cyan", "steelblue"]
        plt.plot(fpr[i], tpr[i], color=color_graph[i],
                label= f'ROC кривая (area = {roc_auc[i]:0.3f}) для {i + 4} соседей')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Пример ROC-кривой')
    plt.legend(loc="lower right")
    plt.show()


def main():
    data = load_breast_cancer()

    X = data.data # Данные
    Y = data.target # Целевые показатели

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state = 42) # Разбиение данных

    scare_minmax = preprocessing.MinMaxScaler()
    # scare_minmax.fit_transform(x_train)
    # scare_minmax.fit_transform(x_test)

    scare_minmax.fit(x_train) # Подготавливает данные к маштабированию
    scare_minmax.transform(x_train) # Маштабирование
    metrics_list = ["euclidean", "manhattan", "chebyshev"]
    for m in metrics_list:
        fprs = []
        tprs = []
        roc_aucs = []
        for n in range(4, 10):
            logreg = KNeighborsClassifier(metric = m, n_neighbors = n) # Тренируем
            logreg.fit(x_train, y_train)
            #answer = logreg.score(x_test, y_test) # Сравнение прогнозов с правильными рез-тами.
            logreg_prob = logreg.predict_proba(x_test)

            logreg_prob = logreg_prob[:,1] # берем только положительные исходы


            # True Positive Rate (TPR) показывает, какой процент среди всех positive верно предсказан моделью.
            # TPR = TP / (TP + FN).

            # False Positive Rate (FPR): какой процент среди всех negative неверно предсказан моделью.
            # FPR = FP / (FP + TN).

            # рассчитываем ROC AUC или area under curve(площадь под графиком)
            lr_auc = roc_auc_score(y_test, logreg_prob)

            # рассчитываем roc-кривую
            fpr, tpr, treshold = roc_curve(y_test, logreg_prob)
            fprs.append(fpr)
            tprs.append(tpr)

            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)
        print(f'ROC for metrics: {m}, neighbours:', *[i for i in range(4, 10)])
        creat_graph(fprs, tprs, roc_aucs)

if __name__ == "__main__":
    main()