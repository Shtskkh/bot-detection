import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from tools.preprocessing import events_in_window, get_column_specifications
    from tools.visualization import plot_column_specs
    import plotly.express as px
    import plotly.graph_objects as go

    _train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    _events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)

    events_train = events_in_window(_train, _events).unique()


@app.cell
def _():
    events_train.head(5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Кардинальность search_query
    """)
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.get_column("search_query")
        .value_counts()
        .sort("count")
        .head(10)
    )
    return


@app.function(hide_code=True)
def top_share(events: pl.DataFrame, col: str, k=5) -> pl.DataFrame:
    _result = events.filter(pl.col(col).is_not_null())

    _result = _result.group_by("target", col).agg(pl.len().alias("count"))

    _result = _result.with_columns(
        (pl.col("count") / pl.col("count").sum().over("target")).alias(
            "share"
        )
    )

    _result = _result.sort(["target", "share"], descending=[False, True])

    _result = _result.group_by("target").head(k)

    _result = _result.with_columns(pl.col(pl.Float64).round(2))

    return _result


@app.cell
def _():
    top_share(events_train, "search_query")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Количество уникальных значений в search_query
    """)
    return


@app.cell
def _():
    events_train.get_column("search_query").n_unique()
    return


@app.cell
def _():
    n_unique_search_query = events_train.filter(pl.col("search_query").is_not_null())

    n_unique_search_query = n_unique_search_query.group_by("cookie_id", "target").agg(
        pl.col("search_query").n_unique().name.prefix("n_unique_")
    )
    return (n_unique_search_query,)


@app.cell(hide_code=True)
def _(n_unique_search_query):
    _specs = get_column_specifications("n_unique_search_query")

    _result = (
        n_unique_search_query.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "n_unique_search_query")
    return


@app.cell(hide_code=True)
def _(n_unique_search_query):
    _fig = px.box(
        n_unique_search_query,
        x="target",
        y="n_unique_search_query",
        color="target",
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение количества уникальных значений в search_query",
            yaxis=go.layout.YAxis(title="Количество уникальных значений"),
            xaxis=go.layout.XAxis(title="Целевая переменная"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Количество null значений в search_query
    """)
    return


@app.cell
def _():
    null_share_of_search_query = events_train.group_by(
        "cookie_id", "target"
    ).agg(pl.col("search_query").is_null().mean().name.prefix("null_share_"))
    return (null_share_of_search_query,)


@app.cell(hide_code=True)
def _(null_share_of_search_query):
    _specs = get_column_specifications("null_share_search_query")

    _result = (
        null_share_of_search_query.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "null_share_search_query")
    return


@app.cell(hide_code=True)
def _(null_share_of_search_query):
    _fig = px.box(
        null_share_of_search_query,
        x="target",
        y="null_share_search_query",
        color="target",
    )

    _fig.update_layout(
        go.Layout(
            title="Распределение null значений в search_query",
            yaxis=go.layout.YAxis(title="Количество null"),
            xaxis=go.layout.XAxis(title="Целевая переменная"),
        )
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Перекрытие search_query
    """)
    return


@app.cell(hide_code=True)
def _():
    _overlap = (
        events_train.filter(pl.col("search_query").is_not_null())
        .group_by("search_query", "target")
        .agg(pl.col("cookie_id").n_unique().alias("n_cookies_same_query"))
    )

    _specs = get_column_specifications("n_cookies_same_query")

    _result = (
        _overlap.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "n_cookies_same_query")
    return


@app.cell(hide_code=True)
def _():
    _df = events_train.with_columns(
        pl.col("event_ts").dt.date().alias("event_date")
    )

    _pool = _df.group_by("event_date").agg(
        pl.col("cookie_id").n_unique().alias("n_cookies_in_day")
    )

    _q_pop = (
        _df.filter(pl.col("search_query").is_not_null())
        .group_by("event_date", "search_query")
        .agg(pl.col("cookie_id").n_unique().alias("n_cookies_same_query"))
        .join(_pool, on="event_date")
        .with_columns(
            (
                pl.col("n_cookies_same_query") / pl.col("n_cookies_in_day")
            ).alias("query_overlap_share")
        )
    )

    _per_cookie = (
        _df.join(
            _q_pop.select("event_date", "search_query", "query_overlap_share"),
            on=["event_date", "search_query"],
            how="left",
        )
        .group_by("cookie_id", "target")
        .agg(
            pl.col("query_overlap_share")
            .mean()
            .alias("avg_query_overlap_share"),
            pl.col("query_overlap_share")
            .max()
            .alias("max_query_overlap_share"),
        )
        .drop_nulls()
    )

    _per_cookie.group_by("target").agg(
        pl.col("avg_query_overlap_share").quantile(0.1).alias("q10"),
        pl.col("avg_query_overlap_share").median().alias("q50"),
        pl.col("avg_query_overlap_share").quantile(0.9).alias("q90"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Параметр `search_query` почти не несёт полезной информации
    """)
    return


if __name__ == "__main__":
    app.run()
