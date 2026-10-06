import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Импорты
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from tools.preprocessing import events_in_window
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return events_in_window, go, make_subplots, mo, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Предпросмотр
    """)
    return


@app.cell
def _(pl):
    train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    test = pl.read_csv("./data/test.csv", try_parse_dates=True)
    events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)
    return events, test, train


@app.cell
def _(train):
    train.head()
    return


@app.cell
def _(test):
    test.head()
    return


@app.cell
def _(events):
    events.head()
    return


@app.cell
def _(events):
    events.schema
    return


@app.cell
def _(events, test, train):
    print(train.shape, test.shape, events.shape)
    return


@app.cell(hide_code=True)
def _(pl, train):
    (
        train.group_by("target")
        .agg(
            pl.len().alias("count"),
            (pl.len() / len(train) * 100).round(2).alias("percent"),
        )
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В train всего 8% ботов.
    """)
    return


@app.cell
def _(events, events_in_window, test, train):
    events_train = events_in_window(train, events)
    events_test = events_in_window(test, events)

    print(events_train.shape, events_test.shape)
    return events_test, events_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Дубликаты
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    def get_duplicate_counts(df: pl.DataFrame) -> pl.DataFrame:
        result = (
            df.is_duplicated().alias("is_duplicate").value_counts(name="count")
        )

        result = result.with_columns(
            (pl.col("count") / len(df) * 100).round(2).alias("percent")
        )

        result = result.sort("is_duplicate")

        return result

    return (get_duplicate_counts,)


@app.cell(hide_code=True)
def _(events, events_test, events_train, get_duplicate_counts, go):
    _events_duplicates = get_duplicate_counts(events)
    _events_train_duplicates = get_duplicate_counts(events_train)
    _events_test_duplicates = get_duplicate_counts(events_test)

    _fig = go.Figure(
        [
            go.Bar(
                name="Процент дубликатов в events",
                x=_events_duplicates.get_column("is_duplicate"),
                y=_events_duplicates.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент дубликатов в events_train",
                x=_events_train_duplicates.get_column("is_duplicate"),
                y=_events_train_duplicates.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент дубликатов в events_test",
                x=_events_test_duplicates.get_column("is_duplicate"),
                y=_events_test_duplicates.get_column("percent"),
                texttemplate="%{y}%",
            ),
        ]
    )

    _fig.update_layout(
        go.Layout(
            title="Процент дубликатов в датасете events",
            xaxis=go.layout.XAxis(
                title="Количество дубликатов", labelalias={"false": "Нет", "true": "Да"}
            ),
            yaxis=go.layout.YAxis(title="% дубликатов", range=[0, 100]),
        ),
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В каждом полученном датасете есть ~3% полностью дублирующихся строк, что может искажать картину.
    """)
    return


@app.cell
def _(events, events_test, events_train):
    unique_events = events.unique()
    unique_ev_tr = events_train.unique()
    unique_ev_te = events_test.unique()

    print(unique_ev_tr.shape, unique_ev_te.shape)
    return unique_ev_te, unique_ev_tr, unique_events


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Пропуски
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    def get_null_counts(df: pl.DataFrame, desc=False) -> pl.DataFrame:
        return (
            df.null_count()
            .transpose(include_header=True, column_names=["null_count"])
            .with_columns(
                (pl.col("null_count") / len(df) * 100)
                .round(2)
                .alias("percent")
            )
            .sort("null_count", descending=desc)
        )

    return (get_null_counts,)


@app.cell(hide_code=True)
def _(get_null_counts, go, unique_ev_te, unique_ev_tr, unique_events):
    _events_null_counts = get_null_counts(unique_events)
    _events_train_null_counts = get_null_counts(unique_ev_tr)
    _events_test_null_counts = get_null_counts(unique_ev_te)

    _fig = go.Figure(
        [
            go.Bar(
                name="Процент пропусков в events",
                x=_events_null_counts.get_column("column"),
                y=_events_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент пропусков в events_train",
                x=_events_train_null_counts.get_column("column"),
                y=_events_train_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент пропусков в events_test",
                x=_events_test_null_counts.get_column("column"),
                y=_events_test_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
        ],
    )

    _fig.update_layout(
        go.Layout(
            title="Процент пропусков в датасете events",
            xaxis=go.layout.XAxis(title="Колонки"),
            yaxis=go.layout.YAxis(title="Процент пропусков"),
        ),
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Распределение событий
    """)
    return


@app.cell(hide_code=True)
def _(go, px, unique_ev_tr):
    ev_name_dist_by_target = px.histogram(
        unique_ev_tr,
        x="event_name",
        histnorm="percent",
        color="target",
        barmode="group",
    )

    ev_name_dist_by_target.update_layout(
        go.Layout(
            title="Распределение событий",
            xaxis=go.layout.XAxis(
                title="Название события", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    ev_name_dist_by_target.update_traces(
        go.Histogram(texttemplate="%{y:.1f}%", textposition="outside")
    )

    ev_name_dist_by_target.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У ботов чаще встречается события ```item_view```, ```search_results_view```. Также, хоть и не значительно, выше ```seller_page_view```, что может говорить нам о том, что боты также сканируют продавцов.

    Люди же при этом чаще делают свайпы и добавляют в избранное.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Распределение количества событий
    """)
    return


@app.cell(hide_code=True)
def _(go, px, unique_ev_tr):
    px.histogram(
        unique_ev_tr, x="target", color="target", text_auto=True
    ).update_layout(
        go.Layout(
            title="Количество событий",
            xaxis=go.layout.XAxis(title="Целевая переменная"),
            yaxis=go.layout.YAxis(
                title="Количество",
            ),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В датасете сильный дисбаланс классов.
    """)
    return


@app.cell(hide_code=True)
def _(pl, unique_ev_tr):
    ev_count = (
        unique_ev_tr.group_by(["cookie_id", "target"])
        .agg(pl.len().alias("events_count"))
        .with_columns(pl.col("events_count").log1p().alias("log_events_count"))
    )
    return (ev_count,)


@app.cell(hide_code=True)
def _(ev_count, px):
    px.histogram(
        ev_count,
        x="events_count",
        color="target",
        barmode="overlay",
        histnorm="probability density",
        nbins=20,
    )
    return


@app.cell(hide_code=True)
def _(ev_count, px):
    px.histogram(
        ev_count,
        x="log_events_count",
        color="target",
        barmode="overlay",
        histnorm="probability density",
        nbins=10,
    )
    return


@app.cell(hide_code=True)
def _(ev_count, pl):
    ev_count_by_target = (
        ev_count.group_by("target")
        .agg(
            pl.col("events_count").mean().alias("mean_events_count"),
            pl.col("events_count").median().alias("median_events_count"),
            pl.col("events_count").std().alias("std_events_count"),
        )
        .with_columns(
            (
                (
                    pl.col("std_events_count")
                    / pl.col("mean_events_count")
                    * 100
                ).alias("cv_events_count")
            )
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )
    return (ev_count_by_target,)


@app.cell(hide_code=True)
def _(ev_count_by_target, go, make_subplots):
    ev_mean_count_dist_by_target_plot = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Среднее",
            "Медиана",
            "Стд. отклонение",
            "Коэфф. вариации",
        ),
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_by_target.get_column("target"),
        y=ev_count_by_target.get_column("mean_events_count"),
        row=1,
        col=1,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_by_target.get_column("target"),
        y=ev_count_by_target.get_column("median_events_count"),
        row=1,
        col=2,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_by_target.get_column("target"),
        y=ev_count_by_target.get_column("std_events_count"),
        row=2,
        col=1,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_by_target.get_column("target"),
        y=ev_count_by_target.get_column("cv_events_count"),
        row=2,
        col=2,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.update_layout(
        go.Layout(
            title="Численные характеристики количества событий",
            showlegend=False,
        )
    )

    ev_mean_count_dist_by_target_plot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Отношения средних к медианам**:

    Боты: ${\dfrac{\bar{X_b}}{\~X_n} \approx \dfrac{29.04}{20} \approx 1.452}$

    Люди: ${\dfrac{\bar{X_h}}{\~X_h} \approx \dfrac{16.62}{12} \approx 1,385}$

    Распределение количества событий скошено вправо по обоим классам, то есть типичный пользователь генерирует мало событий, но есть длинный хвост, который генерирует много событий и тянет среднее вверх.

    **Логарифмирование** ```events_count``` приблизило распределение к симметричному виду, что может быть полезно для линейных моделей (уменьшает влияние выбросов, линеаризует связь с таргетом). При этом стоит учитывать, что для дерево-based моделей (RF/LightGBM), которые мы, скорее всего, будем использовать дальше, монотонное преобразование признака не добавляет разделяющей способности.

    **Коэфф. вариации** показывает, что внутри обоих классов большой разброс поведения.
    """)
    return


@app.cell(hide_code=True)
def _(ev_count, go, px):
    ev_count_box_plot = px.box(
        ev_count,
        x="target",
        y="events_count",
        color="target",
    )

    ev_count_box_plot.update_layout(
        go.Layout(
            title="Распределение количества событий",
            yaxis=go.layout.YAxis(title="Количество событий"),
            xaxis=go.layout.XAxis(title="Целевая переменная"),
        )
    )

    ev_count_box_plot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Почти всегда разница в перцентилях между ботами и людьми держится на уровне ~1.7x, значит это системный сдвиг и боты генерирует в общей массе больше событий.

    Признак ```events_count``` слабый в качестве самостоятельной разделяющей способности (т.к. CV примерно одинаков у обоих классов, а значит внутриклассовый разброс сопоставим с разницей между классами), но, быть может, поведёт себя лучше в купе с другими характеристиками.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Дельты событий
    """)
    return


@app.cell(hide_code=True)
def _(pl, unique_ev_tr):
    ev_deltas = (
        unique_ev_tr.sort(["cookie_id", "event_ts"])
        .with_columns(
            pl.col("event_ts")
            .diff()
            .over("cookie_id")
            .dt.total_seconds()
            .alias("delta_sec")
        )
        .select(["cookie_id", "event_ts", "delta_sec", "target"])
    )
    return (ev_deltas,)


@app.cell(hide_code=True)
def _(ev_deltas, px):
    px.histogram(
        ev_deltas,
        x="delta_sec",
        color="target",
        barmode="group",
        nbins=5,
        log_y=True,
    )
    return


@app.cell(hide_code=True)
def _(ev_deltas, pl):
    ev_deltas_by_target = (
        ev_deltas.group_by("target")
        .agg(
            pl.col("delta_sec").mean().alias("mean_deltas_sec"),
            pl.col("delta_sec").median().alias("median_deltas_sec"),
            pl.col("delta_sec").std().alias("std_deltas_sec"),
        )
        .with_columns(
            (pl.col("std_deltas_sec") / pl.col("mean_deltas_sec") * 100).alias(
                "cv_deltas_sec"
            )
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    ev_deltas_by_target
    return


@app.cell(hide_code=True)
def _(ev_deltas, pl):
    ev_deltas_percentile = (
        ev_deltas.group_by("target")
        .agg(
            pl.col("delta_sec").quantile(q).alias(f"p{q * 100}")
            for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        )
        .sort("target")
    )

    ev_deltas_percentile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    По резкому скачку между p75 и p90 можно предположить, что, что в ```delta_sec``` есть два смешанных процесса: разрывы между сессиями и скорость между событиями в рамках сессии, нужно дальнейшее исследование.

    Но в рамках уже имеющихся данных можно заметить, что интервалы между событиями у ботов в среднем короче.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Посмотрим сессии у ботов при gap в 1800 секунд.
    """)
    return


@app.cell(hide_code=True)
def _(ev_deltas, gap, pl):
    (
        ev_deltas.with_columns(
            pl.col("delta_sec")
            .ge(gap)
            .fill_null(True)
            .cum_sum()
            .over("cookie_id")
            .alias("session")
        )
        .with_columns(
            pl.col("event_ts")
            .diff()
            .dt.total_seconds()
            .over(["cookie_id", "session"])
            .alias("delta_in_session")
        )
        .group_by("target")
        .agg(
            pl.col("session").mean().alias("mean_sessions"),
            pl.col("session").median().alias("median_sessions"),
            pl.col("session").std().alias("std_sessions"),
        )
        .with_columns(
            (pl.col("std_sessions") / pl.col("mean_sessions")).alias(
                "cv_sessions"
            )
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _(ev_deltas, gap, pl):
    (
        ev_deltas.with_columns(
            pl.col("delta_sec")
            .ge(gap)
            .fill_null(True)
            .cum_sum()
            .over("cookie_id")
            .alias("session")
        )
        .with_columns(
            pl.col("event_ts")
            .diff()
            .dt.total_seconds()
            .over(["cookie_id", "session"])
            .alias("delta_in_session")
        )
        .group_by("target")
        .agg(
            pl.col("delta_in_session").mean().alias("mean_delta_in_sessions"),
            pl.col("delta_in_session")
            .median()
            .alias("median_delta_in_sessions"),
            pl.col("delta_in_session").std().alias("std_delta_in_sessions"),
        )
        .with_columns(
            (
                pl.col("std_delta_in_sessions")
                / pl.col("mean_delta_in_sessions")
                * 100
            ).alias("cv_delta_in_sessions")
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Сравним чувствительность gap.
    """)
    return


@app.cell
def _():
    session_gaps = [15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    return (session_gaps,)


@app.cell(hide_code=True)
def _(ev_deltas, pl, session_gaps):
    ev_sessions_by_gap = []

    for gap in session_gaps:
        _ev_sessions = ev_deltas.with_columns(
            pl.col("delta_sec")
            .ge(gap)
            .fill_null(True)
            .cum_sum()
            .over("cookie_id")
            .alias("session")
        ).with_columns(
            pl.col("event_ts")
            .diff()
            .dt.total_seconds()
            .over(["cookie_id", "session"])
            .alias("delta_in_session")
        )

        ev_sessions_by_gap.append(_ev_sessions)
    return ev_sessions_by_gap, gap


@app.cell(hide_code=True)
def _(ev_sessions_by_gap, pl):
    session_gaps_comprassion = pl.DataFrame()

    for _i in ev_sessions_by_gap:
        _grouped = (
            _i.group_by("target")
            .agg(
                pl.col("session").mean().alias("mean_sessions"),
                pl.col("session").median().alias("median_sessions"),
                pl.col("session").std().alias("std_sessions"),
            )
            .with_columns(
                (pl.col("std_sessions") / pl.col("mean_sessions")).alias(
                    "cv_sessions"
                )
            )
            .with_columns(pl.col(pl.Float64).round(2))
            .sort("target")
        )

        session_gaps_comprassion = pl.concat(
            [session_gaps_comprassion, _grouped], how="vertical"
        )

    session_gaps_comprassion
    return


@app.cell(hide_code=True)
def _(ev_sessions_by_gap, pl):
    delta_in_session_comprassion = pl.DataFrame()

    for _i in ev_sessions_by_gap:
        _grouped = (
            _i.group_by("target")
            .agg(
                pl.col("delta_in_session")
                .mean()
                .alias("mean_delta_in_sessions"),
                pl.col("delta_in_session")
                .median()
                .alias("median_delta_in_sessions"),
                pl.col("delta_in_session")
                .std()
                .alias("std_delta_in_sessions"),
            )
            .with_columns(
                (
                    pl.col("std_delta_in_sessions")
                    / pl.col("mean_delta_in_sessions")
                    * 100
                ).alias("cv_delta_in_sessions")
            )
            .with_columns(pl.col(pl.Float64).round(2))
            .sort("target")
        )

        delta_in_session_comprassion = pl.concat(
            [delta_in_session_comprassion, _grouped], how="vertical"
        )

    delta_in_session_comprassion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Очистка от межсессионных разрывов подтвердила, что боты быстрее людей внутри сессии (медиана в ~3 раза меньше при gap=1800), при этом сильно снизила общий шум метрики (std упал на порядок), но одновременно показала, что сама скорость действий у ботов более вариативна относительно своего среднего, чем у людей (cv выше) - возможный признак неоднородности популяции ботов. При этом число сессий на широких gap (900+) как отдельный признак почти не разделяет классы.

    Проверка чувствительности к порогу session gap выявила более сильный паттерн: медиана delta_in_sessions у бота выходит на плато уже к gap≈120-300 сек и дальше почти не растёт (7 -> 13 при расширении gap с 15 до 3600), тогда как у человека медиана продолжает расти вплоть до самых широких порогов (8 -> 41). Это указывает на метрономный, слабо зависящий от ширины окна ритм действий у бота против естественно вариативного поведения человека, где с расширением окна в сессию попадают всё более долгие человеческие паузы (чтение, раздумье). На узких gap (30-60 сек) также заметно расхождение по количеству сессий (~1.7x), которое исчезает на широких порогах — то есть события бота плотнее скучены во времени на короткой шкале.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Кандидаты на включение в итоговые признаки:

    + `median_delta_in_sessions` (при фиксированном gap, например 1800)
    + `std_delta_in_sessions`
    + `cv_delta_in_sessions`
    + `delta_growth` — разница/отношение median_delta между широким и узким gap (например, gap=1800 минус gap=60) как формализация найденного паттерна "плато vs рост"
    + `sessions_count` при узком gap (30-60 сек) — в отличие от широкого gap, здесь есть сигнал
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Количество событий в сессиях
    """)
    return


@app.cell(hide_code=True)
def _(ev_deltas, pl):
    def get_ev_counts_per_session_gap(gap: int) -> pl.DataFrame:
        _result = (
            ev_deltas.with_columns(
                pl.col("delta_sec")
                .ge(gap)
                .fill_null(True)
                .cum_sum()
                .over("cookie_id")
                .alias(f"session_{gap}_sec_gap"),
            )
            .group_by(["cookie_id", f"session_{gap}_sec_gap", "target"])
            .agg(pl.len().alias("events_count_in_session"))
            .group_by(["cookie_id", "target"])
            .agg(
                (
                    pl.col("events_count_in_session").sum()
                    / pl.col(f"session_{gap}_sec_gap").max()
                )
                .round(2)
                .alias(f"events_per_session_{gap}_sec_gap")
            )
            .group_by("target")
            .agg(
                pl.col(f"events_per_session_{gap}_sec_gap")
                .mean()
                .alias(f"mean_events_per_session_{gap}_sec_gap"),
                pl.col(f"events_per_session_{gap}_sec_gap")
                .median()
                .alias(f"median_events_per_session_{gap}_sec_gap"),
                pl.col(f"events_per_session_{gap}_sec_gap")
                .std()
                .alias(f"std_events_per_session_{gap}_sec_gap"),
            )
            .with_columns(
                (
                    pl.col(f"mean_events_per_session_{gap}_sec_gap")
                    / pl.col(f"std_events_per_session_{gap}_sec_gap")
                ).alias(f"events_per_session_{gap}_sec_gap")
            )
            .with_columns(pl.col(pl.Float64).round(2))
            .sort("target")
        )

        return _result

    return (get_ev_counts_per_session_gap,)


@app.cell
def _(get_ev_counts_per_session_gap):
    ev_counts_gap_60 = get_ev_counts_per_session_gap(60)

    ev_counts_gap_60
    return


@app.cell
def _(get_ev_counts_per_session_gap):
    ev_counts_gap_300 = get_ev_counts_per_session_gap(300)

    ev_counts_gap_300
    return


@app.cell
def _(get_ev_counts_per_session_gap):
    ev_counts_gap_1800 = get_ev_counts_per_session_gap(1800)

    ev_counts_gap_1800
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    На узком окне бот делает больше действий в рамках сессии, чем человек, и чем меньше выбранное окно - тем сильнее разница (2.6x при gap=60 против 1.6x при gap=1800). Поэтому для этого признака лучше использовать узкий gap (60 сек), в отличие от delta_in_sessions, где предпочтителен широкий gap.

    CV человека стабильно выше CV бота на всех проверенных порогах - разнородность объёма сессий по количеству действий у человека устойчиво выше, независимо от выбора gap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Кандидаты на включение в итоговые признаки:

    + `events_per_session` (events_count / sessions_count, gap=60) - количество событий на сессию
    + `cv_events_per_session` - устойчив к выбору gap, можно взять на любом (например, том же 1800 для консистентности с другими cv-признаками)
    """)
    return


if __name__ == "__main__":
    app.run()
