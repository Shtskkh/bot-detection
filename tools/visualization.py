import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def plot_column_specs(
    events_aggregated: pl.DataFrame, x_col: str, y_col: str
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Среднее",
            "Медиана",
            "Стд. отклонение",
            "Коэфф. вариации",
        ),
    )

    fig.add_bar(
        x=events_aggregated.get_column(x_col),
        y=events_aggregated.get_column(f"mean_{y_col}"),
        row=1,
        col=1,
        texttemplate="%{y}",
    )

    fig.add_bar(
        x=events_aggregated.get_column(x_col),
        y=events_aggregated.get_column(f"median_{y_col}"),
        row=1,
        col=2,
        texttemplate="%{y}",
    )

    fig.add_bar(
        x=events_aggregated.get_column(x_col),
        y=events_aggregated.get_column(f"std_{y_col}"),
        row=2,
        col=1,
        texttemplate="%{y}",
    )

    fig.add_bar(
        x=events_aggregated.get_column(x_col),
        y=events_aggregated.get_column(f"cv_{y_col}"),
        row=2,
        col=2,
        texttemplate="%{y}",
    )

    fig.update_layout(
        go.Layout(
            title=f"Численные характеристики {y_col}",
            showlegend=False,
        )
    )

    return fig
