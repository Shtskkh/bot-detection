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

    return events_in_window, go, mo, pl


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


if __name__ == "__main__":
    app.run()
