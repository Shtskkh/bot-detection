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
        add_parsed_ua_cols,
        add_platform_match,
    )
    import plotly.express as px
    import plotly.graph_objects as go

    return (
        add_parsed_ua_cols,
        add_platform_match,
        events_in_window,
        go,
        mo,
        pl,
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
    # Распределение платформ
    """)
    return


@app.cell(hide_code=True)
def _(events_train, go, pl, px):
    _fig = px.histogram(
        events_train.with_columns(pl.col("platform").str.to_lowercase()),
        x="platform",
        color="target",
        histnorm="percent",
        barmode="group",
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение платформ",
            xaxis=go.layout.XAxis(
                title="Платформа", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.update_traces(
        go.Histogram(textposition="outside", texttemplate="%{y:0.2f}%")
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для ботов чаще улавливается ```web``` и ```desktop``` как платформа, а также есть разночтение в ```ios``` и ```iphone```.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # user_agent
    """)
    return


@app.cell
def _(add_parsed_ua_cols, events_train, pl):
    events_train_ua = add_parsed_ua_cols(events_train)

    events_train_ua = events_train_ua.with_columns(
        pl.col(["browser", "os", "device"]).fill_null("None")
    )
    return (events_train_ua,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Общие распределения
    """)
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua, x="browser", histnorm="percent", text_auto=True
    )

    _fig.update_layout(
        go.Layout(
            title="Общее распределение браузеров",
            xaxis=go.layout.XAxis(
                title="Браузер", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% от общего"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua, x="os", histnorm="percent", text_auto=True
    )

    _fig.update_layout(
        go.Layout(
            title="Общее распределение операционных систем",
            xaxis=go.layout.XAxis(
                title="Операционная система", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% от общего"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua, x="device", histnorm="percent", text_auto=True
    )

    _fig.update_layout(
        go.Layout(
            title="Общее распределение устройств",
            xaxis=go.layout.XAxis(
                title="Устройство", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% от общего"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Распределения внутри класса
    """)
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua,
        x="browser",
        color="target",
        histnorm="percent",
        barmode="group",
        text_auto=True,
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение браузеров в классе",
            xaxis=go.layout.XAxis(
                title="Браузер", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua,
        x="os",
        color="target",
        histnorm="percent",
        barmode="group",
        text_auto=True,
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение операционных систем в классе",
            xaxis=go.layout.XAxis(
                title="Браузер", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(events_train_ua, go, px):
    _fig = px.histogram(
        events_train_ua,
        x="device",
        color="target",
        histnorm="percent",
        barmode="group",
        text_auto=True,
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение устройств в классе",
            xaxis=go.layout.XAxis(
                title="Браузер", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У ботов чаще встречаются cli инструменты для запросов и Linux как платформа.
    """)
    return


@app.cell(hide_code=True)
def _(events_train_ua, pl):
    (
        events_train_ua.group_by(["cookie_id", "target"])
        .agg(
            pl.col(["platform", "browser", "os"])
            .str.to_lowercase()
            .n_unique(),
        )
        .group_by("target")
        .agg(
            pl.col(["platform", "browser", "os"])
            .mean()
            .name.prefix("mean_")
            .name.suffix("_count")
        )
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Среднее количество платформ, браузеров и операционных систем у ботов и людей совпадает.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Совпадение платформы и операционной системы
    """)
    return


@app.cell(hide_code=True)
def _(add_platform_match, events_train_ua):
    _platform_matches = add_platform_match(
        events_train_ua.get_column("platform").str.to_lowercase(),
        events_train_ua.get_column("os"),
    )

    events_train_ua_match = events_train_ua.with_columns(_platform_matches)
    return (events_train_ua_match,)


@app.cell(hide_code=True)
def _(events_train_ua_match, go, px):
    _fig = px.histogram(
        events_train_ua_match,
        x="platform_match",
        color="target",
        histnorm="percent",
        barmode="group",
        text_auto=True,
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение совпадений платформы и операционной системы",
            xaxis=go.layout.XAxis(
                title="Совпадение", categoryorder="total descending"
            ),
            yaxis=go.layout.YAxis(title="% в классе"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # iphone и ios
    """)
    return


@app.cell(hide_code=True)
def _(events_train_ua, pl):
    (
        events_train_ua.filter(pl.col("platform").is_in(["ios", "iphone"]))
        .group_by("platform")
        .agg(pl.col("os").value_counts())
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Перед обучением нужно свести ios и iphone в одну платформу.
    """)
    return


if __name__ == "__main__":
    app.run()
