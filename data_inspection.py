import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, pl, px


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


@app.cell
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
    # Даты в датасетах
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Дата создания cookie
    """)
    return


@app.cell
def _(train):
    print(
        train.get_column("cookie_created_at").dt.date().min(),
        train.get_column("cookie_created_at").dt.date().max(),
        sep="\n",
    )
    return


@app.cell
def _(test):
    print(
        test.get_column("cookie_created_at").dt.date().min(),
        test.get_column("cookie_created_at").dt.date().max(),
        sep="\n",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Окна наблюдений
    """)
    return


@app.cell
def _(train):
    print(
        train.get_column("window_start_ts").dt.date().min(),
        train.get_column("window_end_ts").dt.date().max(),
        sep="\n",
    )
    return


@app.cell
def _(test):
    print(
        test.get_column("window_start_ts").dt.date().min(),
        test.get_column("window_end_ts").dt.date().max(),
        sep="\n",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # События, в рамках окна наблюдения
    """)
    return


@app.cell
def _(pl):
    def events_in_window(
        events: pl.DataFrame, meta: pl.DataFrame
    ) -> pl.DataFrame:
        return (
            events.join(
                meta,
                on="cookie_id",
                coalesce=True,
            )
            .filter(
                (pl.col("event_ts") >= pl.col("window_start_ts")),
                (pl.col("event_ts") < pl.col("window_end_ts")),
            )
            .drop(["cookie_created_at", "window_start_ts", "window_end_ts"])
        )

    return (events_in_window,)


@app.cell
def _(events, events_in_window, pl, test, train):
    target_mapping = {"0": "Человек", "1": "Бот"}

    events_train = events_in_window(events, train).with_columns(
        pl.col("target").cast(pl.Utf8).replace(target_mapping)
    )
    events_test = events_in_window(events, test)
    return events_test, events_train


@app.cell
def _(events_test, events_train):
    print(events_train.shape, events_test.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Дубликаты
    """)
    return


@app.cell
def _(pl):
    def get_duplicate_counts(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.is_duplicated()
            .alias("is_duplicate")
            .value_counts()
            .with_columns(
                (pl.col("count") / len(df) * 100).round(2).alias("percent")
            )
            .sort("is_duplicate")
        )

    return (get_duplicate_counts,)


@app.cell
def _(events, events_test, events_train, get_duplicate_counts, go):
    ev_train_duplicates_count = get_duplicate_counts(events)
    ev_tr_dublicates_count = get_duplicate_counts(events_train)
    ev_te_dublicates_count = get_duplicate_counts(events_test)

    ev_dupl_counts_bar = go.Figure(
        [
            go.Bar(
                name="Процент дубликатов в events",
                x=ev_train_duplicates_count.get_column("is_duplicate"),
                y=ev_train_duplicates_count.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент дубликатов в events_train",
                x=ev_tr_dublicates_count.get_column("is_duplicate"),
                y=ev_tr_dublicates_count.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент дубликатов в events_test",
                x=ev_te_dublicates_count.get_column("is_duplicate"),
                y=ev_te_dublicates_count.get_column("percent"),
                texttemplate="%{y}%",
            ),
        ],
        layout=go.Layout(
            title="Процент дубликатов в датасете events",
            xaxis=go.layout.XAxis(
                title="Колонки", labelalias={"false": "Нет", "true": "Да"}
            ),
            yaxis=go.layout.YAxis(title="% дубликатов", range=[0, 100]),
        ),
    )

    ev_dupl_counts_bar.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В каждом полученном датасете есть ~3% полностью дублирующихся строк, что может искажать картину. Перед дальнейшим исследованием нужно провести дедупликацию.
    """)
    return


@app.cell(hide_code=True)
def _(events, events_test, events_train):
    unique_events = events.unique()
    unique_ev_tr = events_train.unique()
    unique_ev_te_clean = events_test.unique()

    print(unique_ev_tr.shape, unique_ev_te_clean.shape)
    return unique_ev_te_clean, unique_ev_tr, unique_events


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пропуски
    """)
    return


@app.cell
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
def _(get_null_counts, go, unique_ev_te_clean, unique_ev_tr, unique_events):
    unique_ev_null_counts = get_null_counts(unique_events)
    ev_tr_null_counts = get_null_counts(unique_ev_tr)
    ev_te_null_counts = get_null_counts(unique_ev_te_clean)

    ev_null_counts_bar = go.Figure(
        [
            go.Bar(
                name="Процент пропусков в events",
                x=unique_ev_null_counts.get_column("column"),
                y=unique_ev_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент пропусков в events_train",
                x=ev_tr_null_counts.get_column("column"),
                y=ev_tr_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
            go.Bar(
                name="Процент пропусков в events_test",
                x=ev_te_null_counts.get_column("column"),
                y=ev_te_null_counts.get_column("percent"),
                texttemplate="%{y}%",
            ),
        ],
        layout=go.Layout(
            title="Процент пропусков в датасете events",
            xaxis=go.layout.XAxis(title="Колонки"),
            yaxis=go.layout.YAxis(title="Процент пропусков"),
        ),
    )

    ev_null_counts_bar.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Распределение событий
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
        unique_ev_tr, x="target", color="target", log_y=True, text_auto=True
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
    ev_count_dist_numeric_by_target = (
        unique_ev_tr.group_by(["cookie_id", "target"])
        .agg(pl.len().alias("events_count"))
        .group_by("target")
        .agg(
            (pl.col("events_count").mean()).alias("mean_events_count"),
            (pl.col("events_count").median().alias("median_events_count")),
            (pl.col("events_count").std().alias("std_events_count")),
        )
        .with_columns(
            (
                pl.col("std_events_count") / pl.col("mean_events_count") * 100
            ).alias("cv_events_count")
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )
    return (ev_count_dist_numeric_by_target,)


@app.cell(hide_code=True)
def _(ev_count_dist_numeric_by_target, go, make_subplots):
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
        x=ev_count_dist_numeric_by_target.get_column("target"),
        y=ev_count_dist_numeric_by_target.get_column("mean_events_count"),
        row=1,
        col=1,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_dist_numeric_by_target.get_column("target"),
        y=ev_count_dist_numeric_by_target.get_column("median_events_count"),
        row=1,
        col=2,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_dist_numeric_by_target.get_column("target"),
        y=ev_count_dist_numeric_by_target.get_column("std_events_count"),
        row=2,
        col=1,
        texttemplate="%{y}",
    )

    ev_mean_count_dist_by_target_plot.add_bar(
        x=ev_count_dist_numeric_by_target.get_column("target"),
        y=ev_count_dist_numeric_by_target.get_column("cv_events_count"),
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

    Боты: ${\dfrac{mean_b}{median_b} \approx \dfrac{29.04}{20} \approx 1.452}$

    Люди: ${\dfrac{mean_h}{median_h} \approx \dfrac{16.62}{12} \approx 1,385}$

    Распределение количества событий скошено вправо по обоим классам, то есть типичный пользователь генерирует мало событий, но есть длинный хвост, который генерирует много событий и тянет среднее вверх.

    **Коэфф. вариации** показывает, что внутри обоих классов большой разброс поведения.
    """)
    return


@app.cell(hide_code=True)
def _(go, pl, px, unique_ev_tr):
    ev_count_box_plot = px.box(
        unique_ev_tr.group_by(["cookie_id", "target"]).agg(
            pl.len().alias("events_count")
        ),
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
    ## Распределение часов событий
    """)
    return


@app.cell(hide_code=True)
def _(go, pl, px, unique_ev_tr):
    ev_hour_dist_by_target_plot = px.histogram(
        unique_ev_tr.with_columns(
            pl.col("event_ts").dt.hour().alias("event_hour")
        ),
        x="event_hour",
        color="target",
        histnorm="percent",
        barmode="group",
    )

    ev_hour_dist_by_target_plot.update_layout(
        go.Layout(
            title="Распределение количества событий по времени",
            xaxis=go.layout.XAxis(title="Час дня"),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    ev_hour_dist_by_target_plot.update_traces(
        go.Histogram(textposition="outside", texttemplate="%{y:0.2f}%")
    )

    ev_hour_dist_by_target_plot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Боты заметно активнее в период с 0 до 8 часов, дальше либо больше активность людей, либо паритет.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Распределение платформ
    """)
    return


@app.cell(hide_code=True)
def _(go, pl, px, unique_ev_tr):
    ev_platform_dist_by_target = px.histogram(
        unique_ev_tr.with_columns(pl.col("platform").str.to_lowercase()),
        x="platform",
        color="target",
        histnorm="percent",
        barmode="group",
    )

    ev_platform_dist_by_target.update_layout(
        go.Layout(
            title="Распределение платформ",
            xaxis=go.layout.XAxis(
                title="Платформа", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    ev_platform_dist_by_target.update_traces(
        go.Histogram(textposition="outside", texttemplate="%{y:0.2f}%")
    )

    ev_platform_dist_by_target.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для ботов чаще улавливается ```web``` и ```desktop``` как платформа.
    """)
    return


if __name__ == "__main__":
    app.run()
