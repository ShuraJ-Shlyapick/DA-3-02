import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose as sd


# ---------------------------
# Функции из DA-1-37 и DA-2-04
# ---------------------------

def generate_time_series(start_date: str = "2024-01-01", periods: int = 120, freq: str = "D") -> pd.DataFrame:
    """
    Генерирует синтетический временной ряд с трендом, сезонностью и шумом.

    Параметры
    ----------
    start_date : str
        Начальная дата в формате 'YYYY-MM-DD'.
    periods : int
        Количество временных точек.
    freq : str
        Частота временного ряда (например, 'D' — день).

    Возвращает
    ----------
    pd.DataFrame
        DataFrame с колонками 'date' и 'value'.

    Пример
    -------
    >>> df = generate_time_series(periods=100)
    >>> df.head()
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    t = np.arange(periods)
    # База + линейный тренд + недельная сезонность + шум
    values = (
        100 +                     # базовое значение
        0.2 * t +                 # линейный тренд
        10 * np.sin(2 * np.pi * t / 7) +  # сезонность (7 дней)
        np.random.normal(0, 5, periods)   # случайный шум
    )
    return pd.DataFrame({"date": dates, "value": values})


def seasonal_decompose(
    start_date: str = "2024-01-01",
    periods: int = 100,
    freq: str = "D"
):
    """
    Генерирует временной ряд и выполняет его сезонное разложение.

    Параметры
    ----------
    start_date : str
        Начальная дата ('YYYY-MM-DD').
    periods : int
        Количество временных точек.
    freq : str
        Частота временного ряда.

    Возвращает
    ----------
    statsmodels.tsa.seasonal.DecomposeResult
        Результат декомпозиции (trend, seasonal, resid).

    Примечание
    ----------
    Используется аддитивная модель с периодом 7 (недельная сезонность).
    """
    df = generate_time_series(start_date=start_date, periods=periods, freq=freq)
    result = sd(df['value'], model='additive', period=7)
    return result


def calculation_moving_average(
    df: pd.DataFrame,
    window: int,
    value_col: str,
    new_col: str = None
) -> pd.DataFrame:
    """
    Вычисляет скользящее среднее для указанного столбца.

    Параметры
    ----------
    df : pd.DataFrame
        Исходный DataFrame.
    window : int
        Размер окна для скользящего среднего.
    value_col : str
        Название столбца, по которому вычисляется среднее.
    new_col : str, optional
        Название нового столбца с результатом. По умолчанию — `f"{value_col}_ma{window}"`.

    Возвращает
    ----------
    pd.DataFrame
        DataFrame с добавленным столбцом скользящего среднего.

    Исключения
    ----------
    TypeError
        Если столбец содержит нечисловые значения.

    Пример
    -------
    >>> df = calculation_moving_average(df, window=3, value_col='value')
    """
    if new_col is None:
        new_col = f"{value_col}_ma{window}"

    try:
        numeric_values = pd.to_numeric(df[value_col], errors='raise')
    except ValueError as e:
        raise TypeError(f"Column '{value_col}' contains non-numeric value: {e}")

    df[new_col] = df[value_col].rolling(window=window, min_periods=1).mean()

    return df


# ---------------------------
# Генерация временного ряда
# ---------------------------
df = generate_time_series(periods=120, freq="D")
df.set_index("date", inplace=True)

# ---------------------------
# Скользящее среднее (DA-1-37)
# ---------------------------
df = calculation_moving_average(df, window=3, value_col='value', new_col='rolling_mean')

# ---------------------------
#  Разложение временного ряда (DA-2-04)
# ---------------------------
decomposition = sd(df['value'], model='additive', period=7)  # 7 дней = недельная сезонность
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# ---------------------------
# Прогнозирование (DA-3-02)
# ---------------------------
model = ExponentialSmoothing(
    df['value'],
    trend='add',
    seasonal='add',
    seasonal_periods=7
)
fit = model.fit()
forecast = fit.forecast(steps=5)  # прогноз на 5 шагов вперед

# ---------------------------
# Доверительный интервал (±1.96 * σ остатков)
# ---------------------------
residuals = df['value'] - fit.fittedvalues
sigma = residuals.std()
lower = forecast - 1.96 * sigma
upper = forecast + 1.96 * sigma

# ---------------------------
# Визуализация
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Исходный ряд', color='black', linewidth=1.5)
plt.plot(df.index, df['rolling_mean'], label='Скользящее среднее (3)', linestyle='--', color='blue')
plt.plot(df.index, fit.fittedvalues, label='Сглаженные значения модели', color='green', alpha=0.8)
plt.plot(forecast.index, forecast, label='Прогноз (5 дней)', marker='o', color='red', linewidth=2)

plt.fill_between(forecast.index, lower, upper, color='red', alpha=0.2, label='Доверительный интервал (95%)')

plt.title("Прогноз временного ряда с использованием экспоненциального сглаживания")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# ---------------------------
# Оценка качества прогноза (MAE < 10% от среднего)
# ---------------------------
last_real = df['value'][-5:]      # последние 5 значений из исходного ряда
last_pred = forecast              # прогноз на следующие 5 дней
mae = mean_absolute_error(last_real, last_pred)
mean_val = df['value'].mean()

print(f"MAE: {mae:.1f}")
print(f"Average: {mean_val:.1f}")
print(f"MAE < 10% from average: {'True' if mae < 0.1 * mean_val else 'False'}")