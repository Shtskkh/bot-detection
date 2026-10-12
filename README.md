# Решение задачи по детекции ботов

## Структура директорий

- [data](./data) - данные
- [data_inspection](./data_inspection) - тематические исследования данных по категориям
- [data_inspection_results](./data_inspection_results) - результаты исследований
- [tools](./tools) - переиспользуемые функции

## Текущий результат

| Количество признаков | Precision at recall |
| -------------------- | ------------------- |
| 18                   | 0.7619              |

Требуемый уровень P@R: 0.7.

## Pipeline

Pipeline решения представлен в [pipeline.py](./pipeline.py), а также в [pipyline.html](./pipeline.html) для предпросмотра.

## Использованные библиотеки

- `marimo` - для создания блокнотов
- `polars` - для работы с датасетами
- `plotly` - для визуализации
- `ua_parser` - для парсинга user_agent строки
- `catboost` - для модели классификации
- `optuna` - подбор гиперпараметров

## Источник данных

Avito DataScience Bootcamp 2026
