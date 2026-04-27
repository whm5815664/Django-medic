"""
仪表板数据：User.userID 与 Assessment.userID 对应，
按 assessment_date 取最新一条的 height、weight、bmi、blood_pressure、blood_sugar、health_status。
使用 ORM：先查 User，再按 userID 批量查 Assessment，内存合并。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from django.db.models import OuterRef, Subquery

from main.models import Assessment, User


def _normalize_to_datetime(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return None


def _normalize_to_date(value: date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _calc_age(birth: date | None) -> int | None:
    if birth is None:
        return None
    today = date.today()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


def _fetch_users_for_dashboard() -> list[tuple[Any, ...]]:
    """User ORM：按建档时间倒序，只取展示所需列。"""
    return list(
        User.objects.order_by("-created_at").values_list(
            "userID",
            "userName",
            "userBirth",
            "userTel",
        )
    )


def _latest_assessment_by_user_id(user_ids: list[str]) -> dict[str, dict[str, Any]]:
    """
    Assessment ORM：按 userID + assessment_date 倒序，每个 userID 取第一条。
    如果 assessment_date 已改为 DateTimeField，这里会按最新时分秒取最新记录。
    """
    if not user_ids:
        return {}

    qs = (
        Assessment.objects.filter(userID__in=user_ids)
        .order_by("userID", "-assessment_date")
        .values(
            "userID",
            "height",
            "weight",
            "bmi",
            "blood_pressure",
            "blood_sugar",
            "health_status",
            "assessment_date",
        )
    )

    out: dict[str, dict[str, Any]] = {}
    for row in qs:
        uid = str(row["userID"])
        if uid not in out:
            out[uid] = row
    return out


def _count_latest_unhealthy_users() -> int:
    """
    统计“健康异常人数”：
    - 以 User 为基表（保证人不重复）
    - 对每个 userID 取最新一条 Assessment.health_status
    - 最新检测为“不健康”的用户计数
    """
    latest_status_sq = Subquery(
        Assessment.objects.filter(userID=OuterRef("userID"))
        .order_by("-assessment_date")
        .values("health_status")[:1]
    )
    return User.objects.annotate(latest_health_status=latest_status_sq).filter(latest_health_status="不健康").count()


def build_dashboard_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    查询全部 User（created_at 倒序）；
    对每个 userID 匹配 Assessment.userID，取 assessment_date 最新一条的
    height、weight、bmi、blood_pressure、blood_sugar、health_status。
    """
    user_rows = _fetch_users_for_dashboard()
    user_ids = [str(r[0]) for r in user_rows]
    latest_map = _latest_assessment_by_user_id(user_ids)

    rows: list[dict[str, Any]] = []
    for uid, name, birth, _tel in user_rows:
        uid_s = str(uid)
        latest = latest_map.get(uid_s)

        assessment_dt = _normalize_to_datetime(latest["assessment_date"] if latest else None)
        hs = (latest["health_status"] if latest else None) or ""

        rows.append(
            {
                "pk": uid_s,
                "userID": uid_s,
                "userName": name,
                "age": _calc_age(_normalize_to_date(birth)),
                "height": latest["height"] if latest else None,
                "weight": latest["weight"] if latest else None,
                "bmi": latest["bmi"] if latest else None,
                "blood_pressure": latest["blood_pressure"] if latest else None,
                "blood_sugar": latest["blood_sugar"] if latest else None,
                "health_status": hs,
                "assessment_date_display": assessment_dt.strftime("%Y-%m-%d %H:%M:%S") if assessment_dt else None,
                "updated_at_attr": assessment_dt.strftime("%Y-%m-%d %H:%M:%S") if assessment_dt else "",
            }
        )

    stats = {"user_count": len(rows), "abnormal_count": _count_latest_unhealthy_users()}
    return rows, stats