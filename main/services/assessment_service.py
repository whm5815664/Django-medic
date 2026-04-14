import json
from datetime import date
from decimal import Decimal, InvalidOperation

from main.models import Assessment
from .emotion_analysis import analyze_emotion_logic
from django.utils import timezone

def analyze_and_save_assessment(request_body: bytes | str):
    data = json.loads(request_body or "{}")

    image_data = data.get("image")
    profile = data.get("profile", {})
    user_id = (data.get("userID") or "").strip()
    user_name = (data.get("userName") or "").strip()
    blood_pressure = (data.get("bloodPressure") or "").strip()
    blood_sugar_raw = (data.get("bloodSugar") or "").strip()

    if not image_data:
        raise ValueError("没有图片数据")

    if not user_id:
        raise ValueError("用户ID不能为空")

    age = profile.get("age")
    height = profile.get("height")
    weight = profile.get("weight")

    if age in [None, ""] or height in [None, ""] or weight in [None, ""]:
        raise ValueError("年龄、身高、体重不能为空")

    result = analyze_emotion_logic(
        image_data=image_data,
        profile=profile,
        blood_pressure=blood_pressure,
        blood_sugar=blood_sugar_raw,
    )

    if not result.get("success"):
        raise ValueError(result.get("error") or "分析失败")

    blood_sugar = None
    if blood_sugar_raw != "":
        try:
            blood_sugar = Decimal(blood_sugar_raw)
        except (InvalidOperation, ValueError):
            raise ValueError("血糖格式不正确")

    Assessment.objects.create(
        userID=user_id,
        assessment_date=timezone.now(),
        height=float(height),
        weight=float(weight),
        bmi=float(result.get("bmi") or 0),
        blood_pressure=blood_pressure or None,
        blood_sugar=blood_sugar,
        health_status=result.get("status") or "正常",
    )

    result["userID"] = user_id
    result["userName"] = user_name
    return result