import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from catboost import CatBoostClassifier
    from metric import precision_at_recall

    return CatBoostClassifier, pl, precision_at_recall


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
        _result = meta.join(other=events, on="cookie_id").filter(
            (pl.col("event_ts").ge(pl.col("window_start_ts")))
            & (pl.col("event_ts").lt(pl.col("window_end_ts")))
        )

        return _result

    return (get_events_in_window,)


@app.cell
def _(pl):
    def events_count(events: pl.DataFrame) -> pl.DataFrame:
        return events.group_by("cookie_id").agg(pl.len().alias("events_count"))

    return (events_count,)


@app.cell
def _(events, events_count, get_events_in_window, pl):
    def build_event_features(meta: pl.DataFrame) -> pl.DataFrame:
        out = meta.clone()
        out_events = get_events_in_window(meta, events)

        parts = [events_count(out_events)]

        for p in parts:
            out = out.join(p, on="cookie_id", how="left")

        return out

    return (build_event_features,)


@app.cell
def _(build_event_features, pl, train):
    is_valid = pl.col("window_start_ts").ge(pl.datetime(2026, 4, 17))

    X = build_event_features(train)

    X_train = X.filter(~is_valid)
    X_valid = X.filter(is_valid)

    Xtr = X_train.select(pl.selectors.numeric()).drop("target")
    ytr = X_train.get_column("target")

    Xvl = X_valid.select(pl.selectors.numeric()).drop("target")
    yvl = X_valid.get_column("target")
    return Xtr, Xvl, ytr, yvl


@app.cell
def _(CatBoostClassifier, Xtr, ytr):
    classifier = CatBoostClassifier()

    classifier.fit(Xtr, ytr)
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
