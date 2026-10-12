import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from functools import partial
    from tools.preprocessing import (
        events_in_window,
        add_event_deltas,
        add_sessions,
        add_event_deltas_in_session,
        get_column_specifications,
    )
    from catboost import CatBoostClassifier
    from metric import precision_at_recall

    train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    test = pl.read_csv("./data/test.csv", try_parse_dates=True)
    events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)


@app.function
def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique()


@app.function
def normalize_platform(events: pl.DataFrame) -> pl.DataFrame:
    result = events.with_columns(
        pl.col("platform")
        .str.to_lowercase()
        .replace({"iphone": "ios", "desktop": "web"})
    )

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Базовые признаки
    """)
    return


@app.function
def add_events_count(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.len().alias("events_count")
    )

    return result


@app.function
def add_n_unique_events(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("event_name").n_unique().name.prefix("n_unique_")
    )

    return result


@app.function
def add_item_view_share(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("event_name")
        .eq("item_view")
        .mean()
        .alias("share_of_item_view")
    )

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Временные и сессионные признаки
    """)
    return


@app.function
def add_delta_in_sessions_features(events: pl.DataFrame) -> pl.DataFrame:
    result = events

    steps = [
        add_event_deltas,
        add_sessions,
        add_event_deltas_in_session,
    ]

    for step in steps:
        result = step(result)

    result = result.group_by("cookie_id").agg(
        pl.col("delta_in_session").median().name.prefix("median_"),
        pl.col("delta_in_session").std().name.prefix("std_"),
    )

    return result


@app.function
def add_events_per_session(events: pl.DataFrame) -> pl.DataFrame:
    result = events

    steps = [add_event_deltas, partial(add_sessions, gap=60)]

    for step in steps:
        result = step(result)

    result = (
        result.group_by(["cookie_id", "session"])
        .agg(pl.len().alias("event_count_in_session"))
        .group_by("cookie_id")
        .agg(
            (
                pl.col("event_count_in_session").sum()
                / pl.col("session").max()
            ).alias("events_per_session")
        )
    )

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Платформенные признаки
    """)
    return


@app.function
def add_platform(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("platform").mode().first()
    )

    return result


@app.function
def add_pointer_stats(events: pl.DataFrame) -> pl.DataFrame:
    result = (
        events.filter(pl.col("pointer_x").is_not_null())
        .group_by("cookie_id")
        .agg(
            pl.col("pointer_x", "pointer_y").std().name.prefix("std_"),
            pl.col("pointer_x", "pointer_y")
            .median()
            .name.prefix("median_"),
            (
                pl.col("pointer_x", "pointer_y").n_unique() / pl.len()
            ).name.prefix("unique_ratio_"),
        )
    )

    return result


@app.function
def add_pointers_null_share(
    events: pl.DataFrame,
) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("pointer_x", "pointer_y")
        .is_null()
        .mean()
        .name.prefix("null_share_of_")
    )

    return result


@app.function
def add_web_pointers_null(events: pl.DataFrame) -> pl.DataFrame:
    result = events.filter(pl.col("pointer_x").is_null())

    result = result.group_by("cookie_id").agg(
        pl.col("platform")
        .mode()
        .first()
        .eq("web")
        .alias("web_pointer_miss")
    )

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Признаки предметов
    """)
    return


@app.function
def add_n_unique_item_id(events: pl.DataFrame) -> pl.DataFrame:
    result = events.filter(pl.col("item_id").is_not_null())

    result = result.group_by("cookie_id").agg(
        pl.col("item_id").n_unique().name.prefix("n_unique_")
    )

    return result


@app.function
def add_n_unique_locations(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("item_location").n_unique().name.prefix("n_unique_")
    )

    return result


@app.function
def add_n_unique_categories(events: pl.DataFrame) -> pl.DataFrame:
    result = events.group_by("cookie_id").agg(
        pl.col("item_category").n_unique().name.prefix("n_unqiue_")
    )

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pipeline
    """)
    return


@app.function
def add_features(meta: pl.DataFrame) -> pl.DataFrame:
    out = meta.clone()
    out_events = deduplicate(events_in_window(meta, events))
    out_events = normalize_platform(out_events)

    parts = [
        add_events_count(out_events),
        add_n_unique_events(out_events),
        add_item_view_share(out_events),
        add_delta_in_sessions_features(out_events),
        add_events_per_session(out_events),
        add_platform(out_events),
        add_pointer_stats(out_events),
        add_pointers_null_share(out_events),
        add_web_pointers_null(out_events),
        add_n_unique_item_id(out_events),
        add_n_unique_locations(out_events),
        add_n_unique_categories(out_events),
    ]

    for p in parts:
        out = out.join(p, on="cookie_id", how="left")

    return out


@app.cell
def _():
    DROP_COLS = [
        "cookie_id",
        "cookie_created_at",
        "window_start_ts",
        "window_end_ts",
    ]
    return (DROP_COLS,)


@app.cell
def _(DROP_COLS):
    is_valid = pl.col("window_start_ts").ge(pl.datetime(2026, 4, 17))

    X = add_features(train)

    X_train = X.filter(~is_valid)
    X_valid = X.filter(is_valid)

    Xtr = X_train.drop([*DROP_COLS, "target"])
    ytr = X_train.get_column("target")

    Xvl = X_valid.drop([*DROP_COLS, "target"])
    yvl = X_valid.get_column("target")
    return X, Xtr, Xvl, ytr, yvl


@app.cell
def _(X):
    X.head(5)
    return


@app.cell
def _(Xtr, ytr):
    classifier = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        verbose=500,
        random_state=512,
        auto_class_weights="Balanced",
        cat_features=["platform"],
    )

    classifier.fit(Xtr, ytr)

    feature_importences = pl.DataFrame(
        zip(classifier.feature_names_, classifier.feature_importances_),
        schema=["feature_name", "importance"],
    )

    feature_importences
    return (classifier,)


@app.cell
def _(Xvl, classifier):
    p_val = classifier.predict_proba(Xvl)[:, 1]
    return (p_val,)


@app.cell
def _(p_val, yvl):
    round(precision_at_recall(yvl, p_val), 4)
    return


if __name__ == "__main__":
    app.run()
