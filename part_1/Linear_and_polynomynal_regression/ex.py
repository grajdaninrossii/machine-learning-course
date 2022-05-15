import numpy as np


x = np.array([ 2.45145848, 2.43443939, 4.47361609, 3.01551188, 2.9840347 ,
6.69089459, 4.88697216, 0.69981978, 3.72196012, 5.77547055,
-0.80696851, 0.53496836, -0.61235381, 6.99252145, 5.91239091,
2.50400339, 0.1056696 , 5.60690585, 5.13202216, 4.43408714,
-1.39964518, 4.7448611 , 1.07019687, -1.43211507, 4.82843302,
4.34491461, -0.00898379, 6.25390129, 1.30624911, 3.7397378 ,
5.59952021, 6.78006052, 6.52283222, 3.93814442, 5.29922232,
2.86722983, 1.14810989, 4.42289274, 6.71401969, 1.11832531,
4.21048328, 0.43013272, 3.4466725 , 1.82852476, 1.85961196,
6.4958944 , -0.98625396, 1.93021824, 6.34227227, -0.85632514,
2.07560259, -1.34410788, -1.27084221, 2.76408016, 6.67715051,
5.4918332 , 6.10892948, 3.45547636, 3.52923124, 2.68307057,
5.78001865, 2.35682034, 1.79285104, 4.02607847, 3.05545241,
-0.86956699, 4.99637694, 4.73834678, 3.60904085, 0.44908749,
2.0849003 , 0.93086233, 4.21756339, -1.44753419, 0.09790322,
6.73683231, 4.8304272 , 0.93314062, 6.17818729, 4.06764399,
-1.06890547, 6.76609048, 1.88131007, 6.79566153, 4.55535847,
2.9105759 , 1.34633439, 5.85531874, 5.2269663 , -0.51649885,
5.13810688, 2.13959833, 2.96567234, 4.09157108, 2.84658839,
5.74839385, 4.90374294, 3.43841974, 5.54344208, 1.82181558])

y = np.array([ -6.34484499, -3.63390275, -29.60690453, 1.22236332,
-0.54243886, -32.76041485, -44.33551302, -57.95230234,
-7.71175809, -63.44598511, -21.45104553, -59.77858098,
-36.92064058, -0.33143095, -62.07638965, -3.76209684,
-63.34824989, -62.75411949, -50.61352616, -27.98443057,
65.12179421, -39.57820921, -44.56871081, 69.68587555,
-42.09971363, -26.06101421, -61.92393842, -56.63955343,
-37.19178289, -8.28642144, -63.31460776, -24.37341096,
-43.64609761, -13.23484777, -56.31973808, 0.15434505,
-42.55753008, -27.84946584, -30.9272983 , -44.29714945,
-20.27420899, -59.76409847, -1.33589992, -20.69191481,
-18.48527007, -45.57477731, -0.83128132, -16.58008308,
-54.43098607, -16.91438118, -13.82372801, 53.94138378,
41.68667838, -1.61005506, -33.0531629 , -61.38071825,
-61.02837897, -2.85542735, -3.29001489, -1.60928777,
-63.68860497, -5.38333626, -20.29614338, -15.66764667,
-0.23703505, -13.52504061, -47.72718349, -38.62402827,
-3.66972676, -61.32131336, -12.78127994, -51.3136671 ,
-22.18593952, 73.16754305, -63.5477625 , -29.56491863,
-41.38420395, -51.45566857, -59.13922623, -16.69118026,
8.66532438, -27.65395553, -18.30968592, -22.10451863,
-31.79616253, 1.49292554, -36.11306157, -61.66905453,
-55.29833803, -43.5134154 , -52.57406239, -10.21433731,
-0.93898708, -15.08762017, -0.45652407, -64.88141843,
-44.34735959, -3.10675794, -61.79658631, -19.01354108])

y.shape

import matplotlib.pyplot as plt # Строим графики

ig, axs = plt.subplots(figsize=(10, 10))
plt.title(f'Смотрим точки')
axs.scatter(x, y , color='blue', alpha=0.6)
plt.show()

from sklearn import preprocessing # для нормализации
from sklearn.linear_model import LinearRegression # Класс для осуществления регрессии
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split # Разделение данных на тестовые и тренировочные

poly_regr = PolynomialFeatures(degree = 4)
X_poly = poly_regr.fit_transform(x.reshape(-1, 1)) # Создаем полиноминальные значения

lin_reg = LinearRegression() # Модель, куда подставим полиноминальные значения
lin_reg.fit(X_poly, y)

x_new = np.array([i*(x.max() - x.min())/100 + x.min() for i in np.arange(100)]) # Нормализуем представление

ig, axs = plt.subplots(figsize=(10, 10))
plt.title(f'Полиноминальная регрессия')
axs.scatter(x, y , color='black', alpha=0.6)

# Полиноминальная реггрессия
axs.plot(x_new.reshape(-1, 1), lin_reg.predict(poly_regr.fit_transform(x_new.reshape(-1, 1))), color = "orange")
plt.show()