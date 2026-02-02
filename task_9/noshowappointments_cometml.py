#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек

# In[120]:


import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt


# # Настройка CometML
# <img src='https://camo.githubusercontent.com/b35adf34bc440ea2ac19bb0d3432f48e26d71ab8e4c256e936b709eec44f5b19/68747470733a2f2f7777772e636f6d65742e6d6c2f696d616765732f6c6f676f5f636f6d65745f6c696768742e706e67' width=500px>

# In[121]:


#!pip install comet_ml


# In[122]:


from comet_ml import Experiment


# In[123]:


# Создайте эксперимент с помощью вашего API ключа
experiment = Experiment(
    api_key='99KeV4WB80EqZybAsH3vOJkl8',
    project_name='medical-apointment',
    workspace='yevgeniy-karabekov',
)


# # Загрузка данных

# In[124]:


df = pd.read_csv('./data/KaggleV2-May-2016.csv')


# In[125]:


df.head()


# # Визуализация

# ### Распределение числовых признаков

# In[126]:


df.hist(figsize=(16,14));
#логируем гистограмму
experiment.log_figure(figure=plt)


# *Проанализируем число людей каждого возраста*

# In[127]:


print("Уникальные значения в `Age` => {}".format(df.Age.unique()))


# In[128]:


# Удалим аномальные значения
df = df[(df.Age >= 0) & (df.Age <= 110)]
df.Age.value_counts()


# In[129]:


plt.figure(figsize=(24,6))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Age)
ax.set_title("Распределение пациентов по возрасту")
experiment.log_figure(figure=plt)
plt.show()


# # Придет ли пациент на назначенный прием? 

# In[130]:


df.info()


# ## Кодировка категориальных признаков

# In[131]:


categorical_columns_names = ['Gender', 'Neighbourhood']


# In[132]:


encoded_columns = pd.get_dummies(df, columns = categorical_columns_names, dtype=int)


# In[133]:


encoded_columns.drop(['AppointmentID', 'PatientId', 'ScheduledDay', 'AppointmentDay', 'No-show'], axis=1, inplace = True)


# In[134]:


encoded_columns.head()


# In[135]:


y = df['No-show']


# In[136]:


y


# In[137]:


y = y.replace({'No': 0, 'Yes': 1})


# In[138]:


y.value_counts(True)


# ## Шкалирование данных

# In[139]:


from sklearn.preprocessing import MinMaxScaler


# In[140]:


scaler = MinMaxScaler()
X = scaler.fit_transform(encoded_columns)


# In[141]:


print(X.shape)


# # Модель (этот раздел будет подробно изучен далее в курсе)

# In[142]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ### Инициализация модели

# In[143]:


logreg = LogisticRegression()


# In[144]:


random_state = 42


# ### Обучение модели

# In[145]:


logreg.fit(X, y)


# #### Логирование параметров обучения

# In[ ]:


params={"random_state":random_state,
        "model_type":"logreg"
}


# In[ ]:


experiment.log_parameters(params)


# ### Предсказания модели

# In[ ]:


y_pred = logreg.predict(X)


# ### Анализ модели

# In[ ]:


accuracy = accuracy_score(y, y_pred)

print("Доля правильных ответов: {:6.3f}".format(accuracy))


# # Логирование метрик

# In[ ]:


metrics = {"accuracy":accuracy}


# In[ ]:


experiment.log_metrics(metrics)


# <img src='https://lms.skillfactory.ru/asset-v1:SkillFactory+DST-3.0+28FEB2021+type@asset+block@r2yYhe2DpE4d3WKP.png' width=700px>

# In[ ]:


experiment.log_confusion_matrix(y.tolist(), y_pred.tolist())


# In[ ]:


experiment.display()


# In[ ]:


experiment.end()


# # Домашнее задание

# Визуализируйте корреляцию между числовыми признаками, использованными для обучения модели и залогируйте изображение в Comet.
# 
# В качестве ответа приложите обновленный ноутбук.
# 
# *Форма оценки ментором:*
# 
# 1) Проведена визуализация корреляции между 7 числовыми признаками, использованными для обучения модели - 5 баллов
# 
# 2) Произведено логирование графика в CometMl - 10 баллов
# 
# 
