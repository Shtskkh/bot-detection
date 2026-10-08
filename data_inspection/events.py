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
    from tools.preprocessing import (
        events_in_window,
        get_column_specifications,
        add_event_deltas,
        add_sessions,
        add_event_deltas_in_session,
    )
    from tools.visualization import plot_column_specs
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return (
        add_event_deltas,
        add_event_deltas_in_session,
        add_sessions,
        events_in_window,
        get_column_specifications,
        go,
        mo,
        pl,
        plot_column_specs,
        px,
    )


@app.cell
def _(events_in_window, pl):
    _train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    _events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)

    events_train = events_in_window(_train, _events).unique()
    return (events_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Распределение событий
    """)
    return


@app.cell(hide_code=True)
def _(events_train, go, px):
    _fig = px.histogram(
        events_train,
        x="event_name",
        histnorm="percent",
        color="target",
        barmode="group",
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение событий",
            xaxis=go.layout.XAxis(
                title="Название события", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.update_traces(
        go.Histogram(texttemplate="%{y:.1f}%", textposition="outside")
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У ботов чаще встречается события ```item_view```, ```search_results_view```. Также, хоть и не значительно, выше ```seller_page_view```, что может говорить нам о том, что боты также сканируют продавцов.

    Люди же при этом чаще делают свайпы и добавляют в избранное.
    """)
    return


@app.cell(hide_code=True)
def _(events_train, get_column_specifications, pl, plot_column_specs):
    _result = events_train.group_by("cookie_id", "target").agg(
        pl.col("event_name").n_unique().name.prefix("n_unique_")
    )

    _specs = get_column_specifications("n_unique_event_name")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "n_unique_event_name")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У ботов заметно слабее стандартное отклонение и коэффициент вариации.
    """)
    return


@app.cell(hide_code=True)
def _(events_train, get_column_specifications, pl, plot_column_specs):
    _result = events_train.group_by("cookie_id", "target").agg(
        pl.col("event_name").eq("item_view").mean().alias("share_of_item_view")
    )

    _specs = get_column_specifications("share_of_item_view")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, x_col="target", y_col="share_of_item_view")
    return


@app.cell(hide_code=True)
def _(events_train, get_column_specifications, pl, plot_column_specs):
    _result = events_train.group_by("cookie_id", "target").agg(
        pl.col("event_name").eq("search_results_view").mean().alias("share_of_search_results_view")
    )

    _specs = get_column_specifications("share_of_search_results_view")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, x_col="target", y_col="share_of_search_results_view")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Доля событий `item_view` и `search_results_view` у ботов по коэффициенту вариации сильно стабильнее, чем у людей. Потенциальные признаки.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Распределение количества событий
    """)
    return


@app.cell(hide_code=True)
def _(events_train, go, px):
    px.histogram(
        events_train, x="target", color="target", text_auto=True
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


@app.cell
def _(events_train, pl):
    events_count = (
        events_train.group_by(["cookie_id", "target"])
        .agg(pl.len().alias("events_count"))
        .with_columns(pl.col("events_count").log1p().alias("log_events_count"))
    )
    return (events_count,)


@app.cell(hide_code=True)
def _(events_count, px):
    px.histogram(
        events_count,
        x="events_count",
        color="target",
        barmode="overlay",
        histnorm="probability density",
        nbins=20,
    )
    return


@app.cell(hide_code=True)
def _(events_count, px):
    px.histogram(
        events_count,
        x="log_events_count",
        color="target",
        barmode="overlay",
        histnorm="probability density",
        nbins=10,
    )
    return


@app.cell(hide_code=True)
def _(events_count, get_column_specifications, pl, plot_column_specs):
    _specs = get_column_specifications("events_count")

    _result = (
        events_count.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "events_count")
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
def _(events_count, go, px):
    _fig = px.box(
        events_count,
        x="target",
        y="events_count",
        color="target",
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение количества событий",
            yaxis=go.layout.YAxis(title="Количество событий"),
            xaxis=go.layout.XAxis(title="Целевая переменная"),
        )
    )

    _fig.show()
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
    # Дельты событий
    """)
    return


@app.cell
def _(add_event_deltas, events_train):
    event_deltas = add_event_deltas(events_train)
    return (event_deltas,)


@app.cell(hide_code=True)
def _(event_deltas, px):
    px.histogram(
        event_deltas,
        x="delta_sec",
        color="target",
        barmode="group",
        nbins=5,
        log_y=True,
    )
    return


@app.cell(hide_code=True)
def _(event_deltas, get_column_specifications, pl, plot_column_specs):
    _specs = get_column_specifications("delta_sec")

    _result = (
        event_deltas.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "delta_sec")
    return


@app.cell(hide_code=True)
def _(event_deltas, pl):
    (
        event_deltas.group_by("target")
        .agg(
            pl.col("delta_sec").quantile(q).alias(f"p{q * 100}")
            for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        )
        .sort("target")
    )
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
    # Сессии
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## gap 1800 секунд
    """)
    return


@app.cell
def _(add_sessions, event_deltas):
    _gap = 1800

    sessions_gap_1800 = add_sessions(event_deltas, gap=_gap)
    return (sessions_gap_1800,)


@app.cell(hide_code=True)
def _(get_column_specifications, pl, plot_column_specs, sessions_gap_1800):
    _specs = get_column_specifications("session")

    _result = (
        sessions_gap_1800.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "session")
    return


@app.cell(hide_code=True)
def _(
    add_event_deltas_in_session,
    get_column_specifications,
    plot_column_specs,
    sessions_gap_1800,
):
    _result = add_event_deltas_in_session(sessions_gap_1800)

    _specs = get_column_specifications("delta_in_session")
    _result = _result.group_by("target").agg(_specs).sort("target")

    plot_column_specs(_result, "target", "delta_in_session")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Проверка чувствительности
    """)
    return


@app.cell
def _(add_sessions, event_deltas):
    session_gaps = [15, 30, 60, 120, 300, 600, 900, 1800, 3600]

    sessions_by_gaps = []

    for _gap in session_gaps:
        sessions_by_gaps.append(add_sessions(event_deltas, gap=_gap))
    return session_gaps, sessions_by_gaps


@app.cell(hide_code=True)
def _(get_column_specifications, pl, session_gaps, sessions_by_gaps):
    _comprassion = pl.DataFrame()

    for _gap, _df in zip(session_gaps, sessions_by_gaps):
        _specs = get_column_specifications("session")

        _grouped = (
            _df.group_by("target")
            .agg(_specs)
            .with_columns(pl.col(pl.Float64).round(2))
            .with_columns(pl.lit(_gap).alias("gap"))
            .sort("target")
        )

        _comprassion = pl.concat([_comprassion, _grouped], how="vertical")

    _comprassion
    return


@app.cell(hide_code=True)
def _(
    add_event_deltas_in_session,
    get_column_specifications,
    pl,
    session_gaps,
    sessions_by_gaps,
):
    _comprassion = pl.DataFrame()

    for _gap, _df in zip(session_gaps, sessions_by_gaps):
        _deltas_in_sessions = add_event_deltas_in_session(_df)

        _specs = get_column_specifications("delta_in_session")

        _grouped = (
            _deltas_in_sessions.group_by("target")
            .agg(_specs)
            .with_columns(pl.col(pl.Float64).round(2))
            .with_columns(pl.lit(_gap).alias("gap"))
            .sort("target")
        )

        _comprassion = pl.concat([_comprassion, _grouped], how="vertical")

    _comprassion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Очистка от межсессионных разрывов подтвердила, что боты быстрее людей внутри сессии (медиана в ~3 раза меньше при gap=1800), при этом сильно снизила общий шум метрики (std упал на порядок), но одновременно показала, что сама скорость действий у ботов более вариативна относительно своего среднего, чем у людей (cv выше) - возможный признак неоднородности популяции ботов. При этом число сессий на широких gap (900+) как отдельный признак почти не разделяет классы.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    + `delta_growth` - разница/отношение median_delta между широким и узким gap (например, gap=1800 минус gap=60) как формализация найденного паттерна "плато vs рост"
    + `sessions_count` при узком gap (30-60 сек) — в отличие от широкого gap, здесь есть сигнал
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Количество событий в сессиях
    """)
    return


@app.cell
def _(add_event_deltas, add_sessions, get_column_specifications, pl):
    def get_events_count_per_session_gap(
        events: pl.DataFrame, gap: int
    ) -> pl.DataFrame:

        _result = add_event_deltas(events)
        _result = add_sessions(_result, gap=gap)

        _result = _result.group_by(["cookie_id", "session", "target"]).agg(
            pl.len().alias("events_in_session")
        )

        _result = _result.group_by(["cookie_id", "target"]).agg(
            (
                pl.col("events_in_session").sum() / pl.col("session").max()
            ).alias("events_per_session")
        )

        _specs = get_column_specifications("events_per_session")

        _result = (
            _result.group_by("target")
            .agg(_specs)
            .with_columns(pl.col(pl.Float64).round(2))
            .sort("target")
        )

        return _result

    return (get_events_count_per_session_gap,)


@app.cell
def _(events_train, get_events_count_per_session_gap, plot_column_specs):
    _gap = 60
    _result = get_events_count_per_session_gap(events_train, _gap)

    plot_column_specs(_result, "target", "events_per_session")
    return


@app.cell
def _(events_train, get_events_count_per_session_gap, plot_column_specs):
    _gap = 300
    _result = get_events_count_per_session_gap(events_train, _gap)

    plot_column_specs(_result, "target", "events_per_session")
    return


@app.cell
def _(events_train, get_events_count_per_session_gap, plot_column_specs):
    _gap = 1800
    _result = get_events_count_per_session_gap(events_train, _gap)

    plot_column_specs(_result, "target", "events_per_session")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    На узком окне бот делает больше действий в рамках сессии, чем человек, и чем меньше выбранное окно - тем сильнее разница (2.6x при gap=60 против 1.6x при gap=1800). Поэтому для этого признака лучше использовать узкий gap (60 сек), в отличие от delta_in_sessions, где предпочтителен широкий gap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
