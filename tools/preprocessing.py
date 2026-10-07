import polars as pl
from ua_parser import parse


def events_in_window(
    meta: pl.DataFrame,
    events: pl.DataFrame,
) -> pl.DataFrame:
    """
    Функция получения событий в рамках окна наблюдения. Требование условий.
    """

    # Присоединяем к мета-информации события.
    result = meta.join(events, on="cookie_id", coalesce=True)

    # Фильтруем в рамках окна наблюдения.
    result = result.filter(
        pl.col("event_ts").ge(pl.col("window_start_ts")),
        pl.col("event_ts").lt(pl.col("window_end_ts")),
    )

    # Удаляем мета-информацию.
    result = result.drop(["cookie_created_at", "window_start_ts", "window_end_ts"])

    return result


def get_column_specifications(col: str) -> list[pl.Expr]:
    mean = pl.col(col).mean().name.prefix("mean_")
    median = pl.col(col).median().name.prefix("median_")
    std = pl.col(col).std().name.prefix("std_")
    cv = (pl.col(col).std() / pl.col(col).mean() * 100).alias(f"cv_{col}")

    return [mean, median, std, cv]


def parse_ua_string(ua: str) -> dict:
    parse_result = parse(ua)

    parsed_user_agent = parse_result.user_agent
    browser = parsed_user_agent.family if parsed_user_agent else None

    parsed_os = parse_result.os
    os = parsed_os.family if parsed_os else None

    parsed_device = parse_result.device
    device = parsed_device.family if parsed_device else None

    result = {
        "browser": browser,
        "os": os,
        "device": device,
    }

    return result


def parse_ua_col(ua_col: pl.Series) -> pl.Series:
    parsed_dicts = []

    for ua_str in ua_col:
        parsed_dicts.append(parse_ua_string(ua_str))

    return pl.Series(
        parsed_dicts,
        dtype=pl.Struct(
            {
                "browser": pl.String,
                "os": pl.String,
                "device": pl.String,
            }
        ),
    )


def add_parsed_ua_cols(events: pl.DataFrame) -> pl.DataFrame:
    """
    Добавить колонки browser, os, device из колонки user_agent.
    """
    result = events.with_columns(
        pl.col("user_agent")
        .map_batches(
            parse_ua_col,
            return_dtype=pl.Struct(
                {
                    "browser": pl.String,
                    "os": pl.String,
                    "device": pl.String,
                }
            ),
        )
        .alias("parsed")
    )

    result = result.unnest("parsed")

    return result


def platform_ua_match(platform: str, os: str) -> bool:
    platform = platform.lower()
    os = os.lower()

    if platform == "desktop" and os in ["windows", "mac os x", "linux"]:
        return True

    if platform in ["iphone", "ios"] and os == "ios":
        return True

    if platform == "android" and os == "android":
        return True

    return bool(
        platform == "web" and os in ["windows", "android", "mac os x", "linux", "ios"]
    )


def add_platform_match(platform_col: pl.Series, os_col: pl.Series) -> pl.Series:
    result = []

    for p, o in zip(platform_col, os_col):
        result.append(platform_ua_match(p, o))

    return pl.Series(result, dtype=pl.Boolean).alias("platform_match")


def add_event_deltas(events: pl.DataFrame) -> pl.DataFrame:
    result = events.sort(["cookie_id", "event_ts"]).with_columns(
        pl.col("event_ts")
        .diff()
        .over("cookie_id")
        .dt.total_seconds()
        .alias("delta_sec")
    )

    return result


def add_sessions(events: pl.DataFrame, gap: int = 1800) -> pl.DataFrame:
    result = events.with_columns(
        pl.col("delta_sec")
        .ge(gap)
        .fill_null(True)
        .cum_sum()
        .over("cookie_id")
        .alias("session")
    )

    return result


def add_event_deltas_in_session(events: pl.DataFrame) -> pl.DataFrame:
    result = events.with_columns(
        pl.col("event_ts")
        .diff()
        .dt.total_seconds()
        .over(["cookie_id", "session"])
        .alias("delta_in_session")
    )

    return result
