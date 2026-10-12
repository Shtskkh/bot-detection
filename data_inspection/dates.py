import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from tools.preprocessing import events_in_window

    train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    test = pl.read_csv("./data/test.csv", try_parse_dates=True)
    events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Даты создания cookie
    """)
    return


@app.cell
def _():
    print(
        train.get_column("cookie_created_at").dt.date().min(),
        train.get_column("cookie_created_at").dt.date().max(),
        sep="\n",
    )
    return


@app.cell
def _():
    print(
        test.get_column("cookie_created_at").dt.date().min(),
        test.get_column("cookie_created_at").dt.date().max(),
        sep="\n",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Окна наблюдений
    """)
    return


@app.cell
def _():
    print(
        train.get_column("window_start_ts").dt.date().min(),
        train.get_column("window_end_ts").dt.date().max(),
        sep="\n",
    )
    return


@app.cell
def _():
    print(
        test.get_column("window_start_ts").dt.date().min(),
        test.get_column("window_end_ts").dt.date().max(),
        sep="\n",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Переход в окно наблюдения
    """)
    return


@app.cell
def _():
    events_train = events_in_window(train, events).unique()
    return (events_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Распределение часов событий
    """)
    return


@app.cell(hide_code=True)
def _(events_train):
    _fig = px.histogram(
        events_train.with_columns(pl.col("event_ts").dt.hour().alias("event_hour")),
        x="event_hour",
        color="target",
        histnorm="percent",
        barmode="group",
        nbins=4,
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение количества событий по времени",
            xaxis=go.layout.XAxis(title="Час дня"),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.update_traces(
        go.Histogram(textposition="outside", texttemplate="%{y:0.2f}%")
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Боты заметно активнее в период с 0 до 8 часов, дальше либо больше активность людей, либо паритет, но, даты событий представлены, скорее всего, в UTC формате и без геоданных распределение, скорее всего, не даст нужного сигнала.
    """)
    return


if __name__ == "__main__":
    app.run()
