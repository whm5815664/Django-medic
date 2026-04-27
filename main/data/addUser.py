from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from django.db import IntegrityError, transaction
from django.http import HttpRequest

from main.models import Assessment, User


@dataclass(frozen=True)
class AddUserInput:
    userID: str
    userName: str
    userBirth: date
    userTel: str


def _require(value: str | None, label: str) -> str:
    v = (value or "").strip()
    if not v:
        raise ValueError(f"{label}不能为空")
    return v


def _parse_birth(value: str | None) -> date:
    v = _require(value, "生日")
    try:
        return date.fromisoformat(v)
    except Exception:
        raise ValueError("生日格式不正确，应为 YYYY-MM-DD")


def add_user_from_request(request: HttpRequest) -> dict:
    """
    表单字段：
    - userID: 身份证/用户ID
    - userName: 姓名
    - userBirth: 生日（YYYY-MM-DD）
    - userTel: 电话
    """
    if request.method != "POST":
        raise ValueError("仅支持 POST")

    data = AddUserInput(
        userID=_require(request.POST.get("userID"), "身份证"),
        userName=_require(request.POST.get("userName"), "姓名"),
        userBirth=_parse_birth(request.POST.get("userBirth")),
        userTel=_require(request.POST.get("userTel"), "电话"),
    )

    try:
        with transaction.atomic():
            User.objects.create(
                userID=data.userID,
                userName=data.userName,
                userBirth=data.userBirth,
                userTel=data.userTel,
            )
    except IntegrityError:
        raise ValueError("该身份证已存在，不能重复建档")

    return {"userID": data.userID}


def delete_user_and_assessments(user_id: str) -> dict:
    uid = _require(user_id, "用户ID")
    with transaction.atomic():
        assessments_deleted, _ = Assessment.objects.filter(userID=uid).delete()
        user_deleted, _ = User.objects.filter(userID=uid).delete()
    return {"userID": uid, "deleted_user": user_deleted, "deleted_assessments": assessments_deleted}
