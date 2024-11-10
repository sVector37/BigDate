# -*- coding: utf-8 -*-
# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import scipy
#import h2o
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from pymongo import MongoClient

# Инициализация H2O
h2o.init()
h2o.cluster_info()
h2o.ls()
xlsx = pd.ExcelFile('C:\\data\\db\\modul_2.xlsx')
df = pd.read_excel(xlsx, 'Transactions')
dfnew = pd.read_excel(xlsx, 'NewCustomerList')
dfcust = pd.read_excel(xlsx, 'CustomerDemographic')
dfadr = pd.read_excel(xlsx, 'CustomerAddress')
# Очистка данных
df = df.dropna()
# Проверка и просмотр строк и названий столбцов после импорта
print(df.head())
print(df.columns)
df.describe()
print(dfcust.head())
print(dfcust.columns)
dfcust.describe()
print(dfnew.head())
print(dfnew.columns)
dfnew.describe()
df.count('list_price')

#from pymongo import MongoClient
# Подключение к СУБД
client = MongoClient()
db = client.DWH_bikes
# Формирование коллекций БД и заполнение их соответствующими (отфильтрованными по одному из параметров) фрагментами датафреймов
df_price = df['list_price']
df_cost = df['standard_cost']
df_line = df['product_line']
df_class = df['product_class']
df_size = df['product_size']
df_brand = df['brand']
df_transaction = df['transaction_date']

# Строим графики встроенными возможностями Pandas Basic
df.plot.bar('product_id', 'list_price', color="red")
df.plot.scatter('product_line','product_id', color="red")
df.plot.scatter('product_size','product_id', color="green")
df.plot.scatter('product_class','product_id')

# Воспользуемся возможностями seaborn для анализа
bike = pd.read_excel(xlsx, 'Transactions')
sns.pairplot(bike);
sns.pairplot(data=bike, aspect=.85, hue='brand', size=12);

sns.set(font_scale=1.15)
plt.figure(figsize=(12,6))
sns.heatmap(bike.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1);

# Теперь воспользуемся возможностями matplotlib.pyplot

# Пример: отображение распределения велосипедов по производителям
make_counts = df['brand'].value_counts()
print(make_counts)
# Визуализация распределения велосипедов по производителям
plt.figure(figsize=(12, 6))
plt.title('Распределение велосипедов по производителям', size=12)
plt.xticks(rotation=90)
plt.xlabel('Производитель', size=8)
plt.ylabel('Количество велосипедов', size=12)
plots = sns.barplot(x=make_counts.values, y=make_counts.index, orient='h')
plt.show()

# Оценка средней цены по производителю и классу 
data_df = df.groupby(['brand', 'product_class']).agg(avg_price=('list_price', 'mean'), count=('brand', 'count')) 
data_df = data_df.reset_index() 
print(data_df.head()) 
# Группированная столбчатая диаграмма 
plt.figure(figsize=(12, 6)) 
sns.barplot(x="avg_price", y="brand",hue="brand", data=data_df, palette='Greens', orient='h') 

# Оценка средней цены по производителю и классу 
data2_df = df.groupby(['brand', 'product_line']).agg(avg_price=('list_price', 'mean'), count=('brand', 'count')) 
data2_df = data2_df.reset_index() 
print(data2_df.head()) 
# Группированная столбчатая диаграмма 
plt.figure(figsize=(12, 6)) 
sns.barplot(x="avg_price", y="brand",hue="brand", data=data2_df, orient='h') 

# Оценка средней цены по производителю и классу 
data3_df = df.groupby(['brand', 'product_size']).agg(avg_price=('list_price', 'mean'), count=('brand', 'count')) 
data3_df = data3_df.reset_index()
print(data3_df.head())
# Группированная столбчатая диаграмма 
plt.figure(figsize=(12, 6)) 
sns.barplot(x="avg_price", y="brand",hue="brand", data=data3_df, color="red", orient='h') 

# Оценка средней цены по производителю и линейке 
data_df = df.groupby(['brand', 'product_line']).agg(avg_price=('list_price', 'mean'), count=('brand', 'count')) 
data_df = data_df.reset_index() 
print(data_df.head()) 

# Пример: распределение велосипедов по продуктовой линейке 
linetype_counts = df['product_line'].value_counts()
print(linetype_counts)
# Визуализация распределения велосипедов по 
plt.figure(figsize=(12, 6))
sns.barplot(x=linetype_counts.index, y=linetype_counts.values, color="red")
plt.title('Распределение велосипедов по продуктовой линейке')
plt.xlabel('Продуктовая линейка')
plt.ylabel('Количество велосипедов')
plt.show()

# Пример: распределение велосипедов по размеру
size_counts = df['product_size'].value_counts()
print(size_counts)
# Визуализация распределения велосипедов по размеру
plt.figure(figsize=(12, 6))
sns.barplot(x=size_counts.index, y=size_counts.values)
plt.title('Распределение велосипедов по размеру')
plt.xlabel('Размер')
plt.ylabel('Количество')
plt.show()

# Пример: распределение велосипедов по классу
vclass_counts = df['product_class'].value_counts()
print(vclass_counts)
# Визуализация распределения велосипедов по типу 
plt.figure(figsize=(12, 6))
sns.barplot(x=vclass_counts.index, y=vclass_counts.values, color='Green')
plt.title('Распределение велосипедов по классу')
plt.xlabel('Класс велосипеда')
plt.ylabel('Количество')
plt.show()

# Пример: распределение покупателей по полу
sex_counts = dfcust['gender'].value_counts()
print(sex_counts)
# Визуализация распределения велосипедов по 
plt.figure(figsize=(12, 6))
sns.barplot(x=sex_counts.index, y=sex_counts.values, color='Green')
plt.title('Распределение покупателей по полу')
plt.xlabel('Тип пола')
plt.ylabel('Количество')
plt.show()

# Пример: распределение покупателей по наличию своего авто
car_counts = dfcust['owns_car'].value_counts()
print(car_counts)
# Визуализация распределения велосипедов по 
plt.figure(figsize=(12, 6))
sns.barplot(x=car_counts.index, y=car_counts.values, color='Green')
plt.title('Распределение количества покупок велосипедов у обладателей своего авто')
plt.xlabel('Наличие своего авто')
plt.ylabel('Количество')
plt.show()

# Вычисление возраста покупателей 
def age(born):born = datetime.strptime(born, "%d.%m.%Y").date() 
    today = date.today() 
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day)) 
df['Age'] = df['DOB'].apply(age) 
  
# Вычисление возраста новых покупателей 
def age(born):born = datetime.strptime(born, "%d.%m.%Y").date() 
    today = date.today() 
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day)) 
dfcust['Age'] = dfcust['DOB'].apply(age) 

# Пример: отображение распределения по количеству покупок за 3 года
Z3_counts = dfcust['past_3_years_bike_related_purchases'].value_counts()
print(Z3_counts)
# Визуализация распределения по количеству покупок за 3 года
plt.figure(figsize=(12, 12))
sns.barplot(x=Z3_counts.values, y=Z3_counts.index, palette = 'hls')
plt.xticks(rotation=90)
plt.title('Распределение по количеству покупок за 3 года')
plt.xlabel('Количество покупок запчастей')
plt.ylabel('Количество')
plt.show()

# Пример: отображение распределения покупателей по возрасту
DOB_ocounts = dfcust['Age'].value_counts()
print(DOB_ocounts)
# Визуализация распределения покупателей по возрасту
plt.figure(figsize=(12, 6))
plt.font_scale=0.5
sns.barplot(x=DOB_ocounts.index, y=DOB_ocounts.values, palette = 'hls')
plt.xticks(rotation=90)
plt.title('Распределение покупателей по возрасту')
plt.xlabel('Возраст, лет')
plt.ylabel('Количество')
plt.show()

# Пример: отображение распределения НОВЫХ покупателей по возрасту
DOB_ncounts = dfcust['Age'].value_counts()
print(DOB_ncounts)
# Визуализация распределения покупателейв по возрасту
plt.figure(figsize=(12, 6))
sns.barplot(x=DOB_ncounts.index, y=DOB_ncounts.values, palette = 'hls')
plt.xticks(rotation=90)
plt.title('Распределение покупателей по возрасту')
plt.xlabel('Возраст, лет')
plt.ylabel('Количество')
plt.show()

# Пример: отображение распределения покупателей по обеспеченности
wealth_counts = dfcust['wealth_segment'].value_counts()
print(wealth_counts)
# Визуализация распределения покупателей по 
plt.figure(figsize=(12, 6))
sns.barplot(x=wealth_counts.index, y=wealth_counts.values, color = 'violet', legend='auto')
plt.xticks(rotation=90)
plt.title('Распределение покупателей по обеспеченности')
plt.xlabel('Обеспеченность покупателей')
plt.ylabel('Количество')
plt.show()


# Строим функцию линейной регрессии
from scipy import polyval, stats
fit_output = stats.linregress(df[['customer_id','list_price']])
slope, intercept, r_value, p_value, slope_std_error = fit_output
print(slope, intercept, r_value, p_value, slope_std_error)
# Рисуем график линейной регрессии
import matplotlib.pyplot as plt
plt.plot(df[['customer_id']], df[['list_price']],'o', label='Data')
plt.plot(df[['customer_id']], intercept + slope*df[['list_price']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('Цена продажи, AUD')
plt.xlabel('Покупатель')
plt.legend()
plt.show()

# Теперь воспользуемся возможностями seaborn

# Подготовка данных для кластеризации
X = df[['customer_id', 'list_price']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['customer_id'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='customer_id', y='list_price', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация покупателей по транзакциям за весь год')
plt.xlabel('Покупатель')
plt.ylabel('Цена покупки велосипеда, AUD')
plt.xticks(rotation=90)
plt.show()


# Подготовка данных для кластеризации
X = df[['product_id', 'list_price']]
# Преобразование категориальных данных в числовые
X = pd.get_dummies(X, columns=['product_id'], drop_first=True)
# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Обучение модели кластеризации
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Оценка модели
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {silhouette_avg}')
# Визуализация результатов кластеризации
df['Cluster'] = labels
plt.figure(figsize=(12, 6))
sns.scatterplot(x='product_id', y='list_price', hue='Cluster', data=df, palette='viridis')
plt.title('Кластеризация велосипедов по цене')
plt.xlabel('Модель')
plt.ylabel('Цена покупки велосипеда, AUD')
plt.xticks(rotation=90)
plt.show()