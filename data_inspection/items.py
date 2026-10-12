import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from tools.preprocessing import events_in_window, get_column_specifications
    from tools.visualization import plot_column_specs

    _train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    _events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)

    events_train = events_in_window(_train, _events).unique()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Заполненность
    """)
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.group_by("event_name")
        .agg(
            pl.len().alias("count"),
            pl.col(["item_id", "item_category", "item_location"])
            .is_null()
            .mean()
            .name.prefix("share_null_"),
        )
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("count")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Только на событиях `login` и `search_results_view` item_id всегда пропущен.
    """)
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.group_by("cookie_id", "target")
        .agg(pl.col("item_id").is_not_null().any().alias("has_item"))
        .group_by("target")
        .agg(pl.col("has_item").mean())
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    У ботов всегда есть заполенный item_id, у пользователей он на 0.02 меньше.
    """)
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.select(
            pl.col("item_id").is_null().sum(),
            pl.col("item_category").is_null().sum(),
            pl.col("item_location").is_null().sum()
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Заполненность `item_id`, `item_category` и `item_location` независима друг от друга.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Кардинальность и распределения
    """)
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.select(
            pl.col("item_id", "item_category", "item_location").n_unique().name.prefix("n_unique_")
        )
    )
    return


@app.function
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
    top_share(events_train, "item_category")
    return


@app.cell
def _():
    top_share(events_train, "item_location")
    return


@app.cell(hide_code=True)
def _():
    (
        events_train.filter(pl.col("item_id").is_not_null())
        .group_by("target")
        .agg(
            pl.len().alias("count"),
            pl.col("item_category", "item_location")
            .is_null()
            .mean()
            .name.prefix("null_"),
   
            (
                pl.col("item_category").is_null()
                & pl.col("item_location").is_null()
            )
            .mean()
            .alias("both_null"),
        )
        .with_columns(pl.col(pl.Float64).round(4))
        .sort("target")
    )
    return


@app.cell(hide_code=True)
def _():
    _result = (
        events_train.filter(pl.col("item_id").is_not_null())
        .group_by("cookie_id", "target")
        .agg(pl.col("item_id").n_unique().name.prefix("n_unique_"))
    )

    _specs = get_column_specifications("n_unique_item_id")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "n_unique_item_id")
    return


@app.cell(hide_code=True)
def _():
    _result = (
        events_train.filter(pl.col("item_category").is_not_null())
        .group_by("cookie_id", "target")
        .agg(pl.col("item_category").n_unique().name.prefix("n_unique_"))
    )

    _specs = get_column_specifications("n_unique_item_category")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "n_unique_item_category")
    return


@app.cell(hide_code=True)
def _():
    _result = (
        events_train.filter(pl.col("item_location").is_not_null())
        .group_by("cookie_id", "target")
        .agg(pl.col("item_location").n_unique().name.prefix("n_unique_"))
    )

    _specs = get_column_specifications("n_unique_item_location")

    _result = (
        _result.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
    )

    plot_column_specs(_result, "target", "n_unique_item_location")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    У ботов больше уникальных `item_id` и `item_location`, но различных `item_category` у людей незначительно больше.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Монотонность
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## item_id
    """)
    return


@app.cell
def _():
    item_id_monotonicity = (
        events_train.filter(pl.col("item_id").is_not_null())
        .sort(["cookie_id", "event_ts"])
        .with_columns(
            pl.col("item_id").diff().over("cookie_id").alias("item_id_diff")
        )
        .group_by(["cookie_id", "target"])
        .agg(
            pl.col("item_id_diff")
            .lt(0)
            .mean()
            .alias("pct_increasing_item_id"),
            pl.col("item_id_diff")
            .gt(0)
            .mean()
            .alias("pct_decreasing_item_id"),
        )
        .with_columns(
            pl.max_horizontal(
                ["pct_increasing_item_id", "pct_decreasing_item_id"]
            ).alias("item_id_monotonicity")
        )
    )
    return (item_id_monotonicity,)


@app.cell(hide_code=True)
def _(item_id_monotonicity):
    _specs = get_column_specifications("pct_increasing_item_id")

    _result = (
        item_id_monotonicity.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "pct_increasing_item_id")
    return


@app.cell(hide_code=True)
def _(item_id_monotonicity):
    _specs = get_column_specifications("pct_decreasing_item_id")

    _result = (
        item_id_monotonicity.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "pct_decreasing_item_id")
    return


@app.cell(hide_code=True)
def _(item_id_monotonicity):
    _specs = get_column_specifications("item_id_monotonicity")

    _result = (
        item_id_monotonicity.group_by("target")
        .agg(_specs)
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("target")
    )

    plot_column_specs(_result, "target", "item_id_monotonicity")
    return


if __name__ == "__main__":
    app.run()
