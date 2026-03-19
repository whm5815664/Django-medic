from django.shortcuts import render
from django.conf import settings

from main.SRGA.SRGA_form import srga_submit
from main.SRGA.SRGA_form import srga_reset_temp

def dashboard(request):
    """仪表板页面视图"""
    return render(request, "main/dashboard.html")


def srga_record_form(request):
    """SRGA 健康评估：身高/体重 + 摄像头采集表单页"""
    return render(request, "SRGA/srga_form.html")


def srga_result(request):
    """
    SRGA 推理结果页：根据 user_id 读取 temp 中的图像/音频，并用表单的身高体重（可选年龄）做推理。
    """
    user_id = (request.GET.get("user_id") or "").strip()
    name = (request.GET.get("name") or "").strip()
    collect_time = (request.GET.get("collect_time") or "").strip()

    def _to_float(v, default=0.0):
        try:
            return float(str(v).strip())
        except Exception:
            return float(default)

    height_cm = _to_float(request.GET.get("height_cm"), 0.0)
    weight_kg = _to_float(request.GET.get("weight_kg"), 0.0)
    age = _to_float(request.GET.get("age"), 0.0)

    context = {
        "user_id": user_id,
        "name": name,
        "collect_time": collect_time,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "age": age,
    }

    if not user_id:
        context["error"] = "缺少 user_id，无法推理。请从采集页面提交后进入结果页。"
        return render(request, "SRGA/srga_result.html", context)
    if height_cm <= 0 or weight_kg <= 0:
        context["error"] = "身高/体重不合法（必须为正数），无法推理。"
        return render(request, "SRGA/srga_result.html", context)

    try:
        base_dir = str(settings.BASE_DIR / "main" / "SRGA")
        try:
            from main.SRGA.function import run_srga_inference
        except Exception as e:
            raise RuntimeError(
                f"SRGA 推理依赖未就绪（可能未安装 torch/torchaudio/torchvision 等）。原始错误：{e}"
            )

        result = run_srga_inference(
            base_dir=base_dir,
            user_id=user_id,
            height_cm=height_cm,
            weight_kg=weight_kg,
            age=age,
        )
        context["result"] = result
    except Exception as e:
        context["error"] = f"推理失败：{e}"

    return render(request, "SRGA/srga_result.html", context)

