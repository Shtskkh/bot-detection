import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from functools import partial
    from ua_parser import parse
    from imblearn.over_sampling import SMOTE
    from catboost import CatBoostClassifier
    from metric import precision_at_recall

    return CatBoostClassifier, partial, pl, precision_at_recall


@app.cell
def _(pl):
    train = pl.read_csv("./data/train.csv", try_parse_dates=True)
    test = pl.read_csv("./data/test.csv", try_parse_dates=True)
    events = pl.read_csv("./data/events.csv.gz", try_parse_dates=True)
    return events, train


@app.cell
def _(pl):
    def get_events_in_window(
        meta: pl.DataFrame, events: pl.DataFrame
    ) -> pl.DataFrame:
        result = meta.join(other=events, on="cookie_id").filter(
            (pl.col("event_ts").ge(pl.col("window_start_ts")))
            & (pl.col("event_ts").lt(pl.col("window_end_ts")))
        )

        return result

    return (get_events_in_window,)


@app.cell
def _(pl):
    def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
        return df.unique()

    return (deduplicate,)


@app.cell
def _(pl):
    def add_events_count(events: pl.DataFrame) -> pl.DataFrame:
        result = events.group_by("cookie_id").agg(
            pl.len().alias("events_count")
        )

        return result

    return (add_events_count,)


@app.cell
def _(pl):
    def add_event_deltas(events: pl.DataFrame) -> pl.DataFrame:
        result = (
            events.select(["cookie_id", "event_ts"])
            .sort(["cookie_id", "event_ts"])
            .with_columns(
                pl.col("event_ts")
                .diff()
                .over("cookie_id")
                .dt.total_seconds()
                .alias("delta_sec")
            )
        )

        return result

    return (add_event_deltas,)


@app.cell
def _(pl):
    def add_event_sessions(
        events: pl.DataFrame, gap: int = 1800
    ) -> pl.DataFrame:
        result = events.with_columns(
            pl.col("delta_sec")
            .ge(gap)
            .fill_null(True)
            .cum_sum()
            .over("cookie_id")
            .alias(f"session")
        )

        return result

    return (add_event_sessions,)


@app.cell
def _(pl):
    def add_event_deltas_in_session(events: pl.DataFrame) -> pl.DataFrame:
        result = events.with_columns(
            pl.col("event_ts")
            .diff()
            .dt.total_seconds()
            .over(["cookie_id", "session"])
            .alias("delta_in_session")
        )

        return result

    return (add_event_deltas_in_session,)


@app.cell
def _(add_event_deltas, add_event_deltas_in_session, add_event_sessions, pl):
    def add_median_delta_in_sessions(events: pl.DataFrame) -> pl.DataFrame:
        result = events

        steps = [
            add_event_deltas,
            add_event_sessions,
            add_event_deltas_in_session,
        ]

        for step in steps:
            result = step(result)

        result = result.group_by("cookie_id").agg(
            pl.col("delta_in_session")
            .median()
            .alias("median_delta_in_session"),
            pl.col("delta_in_session").std().alias("std_delta_in_session"),
        )

        return result

    return (add_median_delta_in_sessions,)


@app.cell
def _(add_event_deltas, add_event_sessions, partial, pl):
    def add_events_per_session(events: pl.DataFrame) -> pl.DataFrame:
        result = events

        steps = [add_event_deltas, partial(add_event_sessions, gap=60)]

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

    return (add_events_per_session,)


@app.cell
def _(
    add_events_count,
    add_events_per_session,
    add_median_delta_in_sessions,
    deduplicate,
    events,
    get_events_in_window,
    pl,
):
    def add_features(meta: pl.DataFrame) -> pl.DataFrame:
        out = meta.clone()
        out_events = deduplicate(get_events_in_window(meta, events))

        parts = [
            add_events_count(out_events),
            add_median_delta_in_sessions(out_events),
            add_events_per_session(out_events),
        ]

        for p in parts:
            out = out.join(p, on="cookie_id", how="left")

        return out

    return (add_features,)


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
def _(DROP_COLS, add_features, pl, train):
    is_valid = pl.col("window_start_ts").ge(pl.datetime(2026, 4, 17))

    X = add_features(train)

    X_train = X.filter(~is_valid)
    X_valid = X.filter(is_valid)

    Xtr = X_train.drop([*DROP_COLS, "target"])
    ytr = X_train.get_column("target")

    Xvl = X_valid.drop([*DROP_COLS, "target"])
    yvl = X_valid.get_column("target")
    return Xtr, Xvl, ytr, yvl


@app.cell
def _(CatBoostClassifier, Xtr, pl, ytr):
    classifier = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        verbose=500,
        random_state=512,
        auto_class_weights="Balanced",
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
def _(p_val, precision_at_recall, yvl):
    round(precision_at_recall(yvl, p_val), 4)
    return


if __name__ == "__main__":
    app.run()
