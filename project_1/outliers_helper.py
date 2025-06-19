import numpy as np
import pandas as pd


def outliers_iqr_mod(data, feature, left=1.5, right=1.5, log_scale=False):
    """
    Выявляет выбросы в признаке feature с помощью IQR.

    Параметры:
    - data: DataFrame
    - feature: строка, имя признака
    - left, right: множители IQR влево и вправо
    - log_scale: если True, используется логарифм признака (только для положительных значений)

    Возвращает:
    - outliers: выбросы
    - cleaned: данные без выбросов
    - outlier_indices: индексы выбросов
    """
    x = data[feature].copy()

    if log_scale:
        x = x[x > 0]
        x = np.log(x)

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - left * iqr
    upper_bound = q3 + right * iqr

    mask_outliers = (x < lower_bound) | (x > upper_bound)
    outlier_indices = x[mask_outliers].index

    outliers = data.loc[outlier_indices, [feature]]
    cleaned = data.drop(index=outlier_indices)[[feature]]

    return outliers, cleaned, outlier_indices




def outliers_z_score_mod(data, feature, log_scale=False, left=3, right=3):
    """
    Выявляет выбросы по признаку feature в датафрейме data с помощью Z-отклонения.

    Параметры:
    - data: pandas DataFrame или путь к CSV-файлу
    - feature: имя числового столбца
    - log_scale: логарифмировать ли данные перед анализом (log1p)
    - left: число стандартных отклонений влево от среднего
    - right: число стандартных отклонений вправо от среднего

    Возвращает:
    - outliers: выбросы
    - cleaned: данные без выбросов
    - outlier_indices: индексы выбросов
    """

    if isinstance(data, str):
        data = pd.read_csv(data)

    x = data[feature].copy()

    if log_scale:
        x = x[x >= 0]
        x = np.log1p(x)

    mean = x.mean()
    std = x.std()

    lower_bound = mean - left * std
    upper_bound = mean + right * std

    mask_outliers = (x < lower_bound) | (x > upper_bound)
    outlier_indices = x[mask_outliers].index

    outliers = data.loc[outlier_indices, [feature]]
    cleaned = data.drop(index=outlier_indices)[[feature]]

    return outliers, cleaned, outlier_indices
