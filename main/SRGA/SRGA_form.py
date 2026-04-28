import base64
import csv
import os
import shutil
from django.http import JsonResponse
from django.conf import settings
from django.utils.http import urlencode


def _strip_data_url_prefix(b64_or_data_url: str) -> str:
    """支持传入 data:*/*;base64,xxxx 或直接的 b64 内容"""
    if not b64_or_data_url:
        return ""
    if "," in b64_or_data_url:
        return b64_or_data_url.split(",", 1)[1]
    return b64_or_data_url


def srga_submit(request):
    """
    接收前端提交的最后一帧图像 + 最后一段音频（wav）
    并将其保存到 `main/SRGA/temp`，便于后续用 function.py 读取。
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

    try:
        height_cm = float(request.POST.get("height_cm", "").strip())
        weight_kg = float(request.POST.get("weight_kg", "").strip())
        age = float(request.POST.get("age", "0").strip() or "0")
        name = request.POST.get("name", "").strip()
        user_id = request.POST.get("user_id", "").strip()
        collect_time = request.POST.get("collect_time", "").strip()

        last_frame_image_b64 = request.POST.get("last_frame_image_b64", "").strip()
        last_audio_wav_b64 = request.POST.get("last_audio_wav_b64", "").strip()

        if height_cm <= 0 or weight_kg <= 0:
            return JsonResponse({"status": "error", "message": "身高/体重必须为正数"}, status=400)
        if not name:
            return JsonResponse({"status": "error", "message": "姓名不能为空"}, status=400)
        if not user_id:
            return JsonResponse({"status": "error", "message": "用户编号不能为空（user_id）"}, status=400)
        if not collect_time:
            return JsonResponse({"status": "error", "message": "采集时间不能为空"}, status=400)
        if not last_frame_image_b64 or not last_audio_wav_b64:
            return JsonResponse({"status": "error", "message": "缺少图像或音频数据"}, status=400)

        uid = user_id

        # 先把前端截取的“最后一帧图像/最后音频”暂存到 temp 目录
        dataset_dir = os.path.join(settings.BASE_DIR, "main", "SRGA", "temp")
        images_dir = os.path.join(dataset_dir, "images")
        audios_dir = os.path.join(dataset_dir, "audios")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(audios_dir, exist_ok=True)

        # 保存图像
        img_bytes = base64.b64decode(_strip_data_url_prefix(last_frame_image_b64))
        img_path = os.path.join(images_dir, f"{uid}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # 保存音频（wav）
        audio_bytes = base64.b64decode(_strip_data_url_prefix(last_audio_wav_b64))
        audio_path = os.path.join(audios_dir, f"{uid}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 写入/追加 tabular.csv（function.py 预处理器会读取）
        tabular_path = os.path.join(dataset_dir, "tabular.csv")
        expected_header = ["user_id", "weight(kg)", "height(cm)", "age"]

        # 若历史文件表头/列数不一致，会导致 pandas 读取时报 “Expected N fields...”
        # 这里统一保证 tabular.csv 永远只有 4 列，且表头固定。
        mode = "a"
        need_header = True
        if os.path.exists(tabular_path) and os.path.getsize(tabular_path) > 0:
            try:
                with open(tabular_path, "r", newline="", encoding="utf-8") as rf:
                    reader = csv.reader(rf)
                    first_row = next(reader, [])
                if first_row == expected_header:
                    need_header = False
                else:
                    mode = "w"  # 覆盖为新格式，避免后续追加造成列数混乱
                    need_header = True
            except Exception:
                mode = "w"
                need_header = True
        else:
            mode = "w"
            need_header = True

        with open(tabular_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if need_header:
                writer.writerow(expected_header)
            writer.writerow([uid, weight_kg, height_cm, age])

        result_url = "/srga/result/?" + urlencode(
            {
                "user_id": uid,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "age": age,
                "name": name,
                "collect_time": collect_time,
            }
        )

        return JsonResponse(
            {
                "status": "ok",
                "message": "已接收并保存采集数据到 SRGA/temp（最后一帧图像与最后音频）。后续可基于填写的 user_id 进行 SRGA 推理。",
                "uid": uid,
                "result_url": result_url,
                "saved": {
                    "image": os.path.relpath(img_path, settings.BASE_DIR),
                    "audio": os.path.relpath(audio_path, settings.BASE_DIR),
                    "tabular": os.path.relpath(tabular_path, settings.BASE_DIR),
                },
            }
        )
    except base64.binascii.Error:
        return JsonResponse({"status": "error", "message": "base64 数据解码失败（图像/音频可能被截断）"}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"服务器处理失败：{e}"}, status=500)


def srga_reset_temp(request):
    """
    页面打开时重置 `main/SRGA/temp`：
    - 删除 tabular.csv
    - 清空 audios、images 目录下所有文件
    - 重新创建 audios/images 目录
    """
    if request.method not in ("POST", "GET"):
        return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)

    try:
        dataset_dir = os.path.join(settings.BASE_DIR, "main", "SRGA", "temp")
        images_dir = os.path.join(dataset_dir, "images")
        audios_dir = os.path.join(dataset_dir, "audios")
        tabular_path = os.path.join(dataset_dir, "tabular.csv")

        os.makedirs(dataset_dir, exist_ok=True)

        # 删除 tabular.csv
        if os.path.exists(tabular_path):
            try:
                os.remove(tabular_path)
            except Exception:
                pass

        # 重建 images / audios（删除整个目录再创建）
        for d in (images_dir, audios_dir):
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    # rmtree 失败则退化为逐文件删除
                    try:
                        for name in os.listdir(d):
                            p = os.path.join(d, name)
                            try:
                                if os.path.isfile(p) or os.path.islink(p):
                                    os.remove(p)
                                elif os.path.isdir(p):
                                    shutil.rmtree(p)
                            except Exception:
                                pass
                    except Exception:
                        pass
            os.makedirs(d, exist_ok=True)

        return JsonResponse(
            {
                "status": "ok",
                "message": "SRGA/temp 已重置（tabular.csv、images、audios 已清空并重建）。",
            }
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"重置失败：{e}"}, status=500)