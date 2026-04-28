import json
import time
from typing import Any, Dict, Optional, Generator

import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .ollama_config import OPENCODE_BASE_URL, OPENCODE_MODEL

OPENCODE_BASE_URL = OPENCODE_BASE_URL
model = OPENCODE_MODEL

#model = {'model': 'Big Pickle', 'modelID': 'big-pickle', 'providerID': 'opencode'}
#model = {'model': 'glm-4.7-flash:latest', 'modelID': 'glm-4.7-flash:latest', 'providerID': 'ollama'}
#model = {'model': 'gpt-oss:latest', 'modelID': 'gpt-oss:latest', 'providerID': 'ollama'}

# 列出所有会话
def get_session(base_url: str) -> list:
    r = requests.get(f"{base_url}/session")
    sessions = r.json()
    return [
        {'id': s.get('id'), 'title': s.get('title'), 'directory': s.get('directory')}
        for s in sessions
    ]


# 删除会话
def delete_session(base_url: str, session_id: str) -> Dict[str, Any]:
    r = requests.delete(f"{base_url}/session/{session_id}")
    return r.json()


# 智能体角色设定
AGENT_SYSTEM_PROMPT = """
你是Aiot团队开发的健康评估智能体
系统的开发者为：WHM
"""
AGENT_SYSTEM_TEMP = r"Django-dashboard\aiModels\agent\temp"


# 创建会话
def creat_session(base_url: str, title: str = "智能体助手") -> Dict[str, Any]:
    r = requests.post(f"{base_url}/session", json={"title": title})
    session = r.json()
    session_id = session.get("id")
    print('agent创建会话id:', session_id)
    
    # 创建会话后立即加载角色设定
    if session_id:
        init_msg = f"""请加载以下角色设定：\n{AGENT_SYSTEM_PROMPT}"""
        send_async_message(init_msg, base_url, session_id, model_config=model, no_reply=True)
    
    return session



# ---------- Django 视图：供 agent.html 调用 ----------

@csrf_exempt
@require_POST
def agent_create_session_view(request):
    """创建 opencode 会话，页面打开时调用。"""
    try:
        data = json.loads(request.body) if request.body else {}
        title = data.get("title", "智能体助手")
        session = creat_session(OPENCODE_BASE_URL, title=title)
        # session_id 取自 opencode 返回的 json['id']
        session_id = session.get("id") if isinstance(session, dict) else None
        if not session_id:
            return JsonResponse({"success": False, "error": "opencode 未返回 session id"})
        return JsonResponse({"success": True, "session_id": session_id})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})




@csrf_exempt
@require_POST
def agent_delete_session_view(request):
    """删除 opencode 会话，页面关闭时调用。"""
    try:
        data = json.loads(request.body) or {}
        session_id = data.get("session_id")
        if not session_id:
            return JsonResponse({"success": False, "error": "缺少 session_id"})
        delete_session(OPENCODE_BASE_URL, session_id)
        return JsonResponse({"success": True})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})



# ------新的异步消息方法-------------

def send_async_message(
    message: str,
    base_url: str,
    session_id: str,
    agent: Optional[str] = "general", 
    model_config: Optional[Dict[str, Any]] = model,
    no_reply: bool = False
):
    """异步发送消息到 opencode 会话"""
    data = {
        "parts": [{"type": "text", "text": message}]
    }
    if agent:
        data["agent"] = agent
    if model_config:
        data["model"] = model_config
    if no_reply:
        data["noReply"] = True
    
    r = requests.post(
        f"{base_url}/session/{session_id}/prompt_async",
        json=data,
        timeout=50
    )
    print("消息已发送")
    return r


def _get_latest_user_message_id(base_url: str, session_id: str, retries: int = 5, sleep_s: float = 0.2) -> Optional[str]:
    """获取最近一条 user message id（用于过滤本次回复）。"""
    try:
        for _ in range(max(1, int(retries))):
            msgs_resp = requests.get(
                f"{base_url}/session/{session_id}/message",
                timeout=10,
            )
            msgs = msgs_resp.json()
            for m in reversed(msgs):
                info = m.get("info", {})
                if info.get("role") == "user":
                    mid = info.get("id")
                    if mid:
                        return mid
            time.sleep(max(0.0, float(sleep_s)))
    except Exception as e:
        print("获取当前 user message 失败：", e)
    return None


def _build_sse_response(base_url: str, session_id: str, target_parent_id: Optional[str]) -> StreamingHttpResponse:
    def event_stream():
        reasoning_content = ""
        text_content = ""

        for chunk in stream_output(
            base_url,
            session_id,
            interval=0.5,
            parent_message_id=target_parent_id,
        ):
            if chunk["type"] == "reasoning":
                reasoning_content += chunk["content"]
                yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk['content']}, ensure_ascii=False)}\n\n"

            elif chunk["type"] == "text":
                text_content += chunk["content"]
                yield f"data: {json.dumps({'type': 'text', 'content': chunk['content']}, ensure_ascii=False)}\n\n"

            elif chunk["type"] == "finished":
                if not text_content.strip():
                    if reasoning_content:
                        text_content = "（本轮已完成推理与工具调用，未生成额外文字回复；详见上方推理过程。）"
                    else:
                        text_content = "（本轮仅执行了工具调用，无文字回复。）"
                yield f"data: {json.dumps({'type': 'finished', 'reasoning': reasoning_content, 'text': text_content}, ensure_ascii=False)}\n\n"
                break

            elif chunk["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': chunk['content']}, ensure_ascii=False)}\n\n"
                break

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def send_message_sse(session_id: str, message: str) -> StreamingHttpResponse:
    """统一入口：发送 message 到会话，并以 SSE 返回本次输出。"""
    send_async_message(message, OPENCODE_BASE_URL, session_id, model_config=model)
    target_parent_id = _get_latest_user_message_id(OPENCODE_BASE_URL, session_id)
    return _build_sse_response(OPENCODE_BASE_URL, session_id, target_parent_id)


def stream_output(
    base_url: str,
    session_id: str,
    interval: float = 5.0,
    parent_message_id: Optional[str] = None,
    max_stream_seconds: float = 300.0,
) -> Generator[Dict[str, Any], None, None]:
    """流式获取输出，返回生成器，每次 yield 一个包含 type 和 content 的字典。

    注意：
    - opencode 会对同一条 assistant message 进行"就地更新"，同一个 part 的 text 会从空字符串逐步补全。
    - 这里使用 part_id -> 已打印字符长度 的映射，每次只输出新增的部分。
    - finish / step-finish 为 tool-calls 时不结束流，继续轮询直至模型在工具后生成文字或超时。
    """
    printed_text_lens: Dict[str, int] = {}  # {part_id: 已打印的字符长度}
    printed_tool_ids = set()  # 已处理的 question 等工具 part（避免重复）
    tool_stdout_emitted = set()  # 已推送 stdout 的 bash/shell part（需在 status=completed 时再推）
    last_message_id = None
    stream_started = time.time()
    last_delta_ts = time.time()
    pending_finish = False
    pending_finish_since: Optional[float] = None

    # 当检测到 finish 时，不立刻结束；需要满足“输出已稳定不再增长”才发 finished 给前端。
    # 这是为了避免 opencode 在 parts 仍在就地补全时提前写入 info.finish，导致前端过早停止读流。
    finish_stable_seconds = 1.0

    while True:
        try:
            if time.time() - stream_started > max_stream_seconds:
                yield {"type": "error", "content": "流式输出超时，请重试或缩短问题。"}
                break

            r = requests.get(
                f"{base_url}/session/{session_id}/message",
                timeout=30
            )
            messages = r.json()
            print('messages:', messages)
            if messages:
                tail = messages[-1]
                tinfo = tail.get("info", {})
                print(
                    f"SSE poll: role={tinfo.get('role')} id={tinfo.get('id')} finish={tinfo.get('finish')}",
                    flush=True,
                )

            if not messages:
                time.sleep(interval)
                continue

            # 找最后一条 assistant 消息
            assistant_msg = None
            for msg in reversed(messages):
                if msg.get("info", {}).get("role") == "assistant":
                    assistant_msg = msg
                    break

            if not assistant_msg:
                time.sleep(interval)
                continue

            # 如果指定了要跟踪的 parent_message_id，则只处理与当前问题对应的回复
            if parent_message_id:
                parent_id = assistant_msg.get("info", {}).get("parentID")
                if parent_id != parent_message_id:
                    # 这条 assistant 回复不是当前问题的回答，跳过
                    time.sleep(interval)
                    continue

            message_id = assistant_msg["info"]["id"]

            # 如果进入了一条新的 assistant 消息，清空已打印记录
            if message_id != last_message_id:
                printed_text_lens = {}
                printed_tool_ids = set()
                tool_stdout_emitted = set()
                last_message_id = message_id
                last_delta_ts = time.time()
                pending_finish = False
                pending_finish_since = None
                print(f"\n===== assistant message: {message_id} =====", flush=True)

            finished = False
            had_delta_this_poll = False

            for part in assistant_msg.get("parts", []):
                part_id = part.get("id")
                part_type = part.get("type")

                if not part_id:
                    continue

                # 只对 reasoning/text 做增量输出
                if part_type in ("reasoning", "text"):
                    current_text = part.get("text", "") or ""
                    old_len = printed_text_lens.get(part_id, 0)

                    # 只打印新增部分
                    if len(current_text) > old_len:
                        delta = current_text[old_len:]

                        # 第一次打印这个 part 时，先打印标签
                        if old_len == 0:
                            if part_type == "reasoning":
                                print(f"\n[reasoning] ", end="", flush=True)
                            else:
                                print(f"\n[text] ", end="", flush=True)

                        print(delta, end="", flush=True)
                        printed_text_lens[part_id] = len(current_text)
                        had_delta_this_poll = True
                        last_delta_ts = time.time()
                        
                        # yield 给前端
                        yield {"type": part_type, "content": delta}

                #工具调用标识
                elif part_type == "tool":
                    tool_name = part.get("tool")
                    state = part.get("state", {}) or {}
                    status = state.get("status")

                    # bash 首次变为 completed 时再推 stdout（此前轮询可能为 running，不能用 printed_tool_ids 提前跳过）
                    if status == "completed" and tool_name in ("bash", "shell"):
                        if part_id not in tool_stdout_emitted:
                            tool_stdout_emitted.add(part_id)
                            meta = state.get("metadata") or {}
                            raw_out = meta.get("output") or state.get("output") or ""
                            if isinstance(raw_out, str) and raw_out.strip():
                                snippet = raw_out.strip()
                                if len(snippet) > 12000:
                                    snippet = snippet[:12000] + "\n…(输出已截断)"
                                block = f"\n\n【命令输出】\n{snippet}\n"
                                # 需求：当工具 status==completed 时，将“命令输出”放到 reasoning 栏位显示
                                print("\n[reasoning] " + block[:500] + ("…" if len(block) > 500 else ""), flush=True)
                                yield {"type": "reasoning", "content": block}

                    if tool_name == "question" and status in ("running", "pending"):
                        if part_id in printed_tool_ids:
                            continue
                        printed_tool_ids.add(part_id)
                        questions = state.get("input", {}).get("questions", []) or []
                
                        text = "我需要先确认以下信息，才能继续：\n"
                        for i, q in enumerate(questions, 1):
                            text += f"\n{i}. {q.get('question', q.get('header', f'问题{i}'))}"
                            options = q.get("options", []) or []
                            if options:
                                text += "\n   可选项：" + " / ".join(
                                    opt.get("label", "") for opt in options
                                )
                
                        print("\n[text] " + text, flush=True)
                        return
                    
                # 结束表示（工具调用后的 step-finish 不应结束 SSE，需等下一轮 assistant）
                elif part_type == "step-finish":
                    info_f = assistant_msg.get("info", {}).get("finish")
                    if part.get("reason") == "tool-calls" or info_f == "tool-calls":
                        pass
                    else:
                        finished = True

            info_finish = assistant_msg.get("info", {}).get("finish")
            # tool-calls 表示仍可能继续生成，不可在此处结束流
            if not finished and info_finish and info_finish != "tool-calls":
                finished = True

            # 看到 finished 先进入“待结束”状态，避免最后一段增量还没轮询到就提前 break
            if finished and not pending_finish:
                pending_finish = True
                pending_finish_since = time.time()

            # 若已进入待结束状态：需要满足一段时间内没有任何增量输出，才真正结束 SSE
            if pending_finish:
                now = time.time()
                since_delta = now - last_delta_ts
                since_finish = now - (pending_finish_since or now)
                if since_delta >= finish_stable_seconds and since_finish >= finish_stable_seconds:
                    yield {"type": "finished", "content": ""}
                    break

        except Exception as e:
            print("查询失败：", e, flush=True)
            yield {"type": "error", "content": str(e)}
            break

        time.sleep(interval)


@csrf_exempt
@require_POST
def agent_send_message_view(request):
    """发送消息到 opencode 会话，使用流式输出（SSE）"""
    try:
        data = json.loads(request.body) or {}
        session_id = data.get("session_id")
        message = data.get("message", "").strip()
        if not session_id:
            return JsonResponse({"success": False, "error": "缺少 session_id"})
        if not message:
            return JsonResponse({"success": False, "error": "消息不能为空"})
        return send_message_sse(session_id, message)
        
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})