# Django-medic

## 系统简介
`Django-medic` 是一个基于 **Django** 的医疗场景 Web 系统示例工程，提供基础的站点框架（项目配置、路由、模板页面）以及一个面向多模态输入的 **SRGA** 推理/演示模块。系统目标是将“表单提交 → 后端处理 → 结果展示”的完整链路串起来，便于在此基础上扩展业务功能与算法能力。

## 核心功能
- **Web 端页面**：提供仪表盘（Dashboard）与 SRGA 结果展示页面等模板。
- **SRGA 模块集成**：支持多模态数据（如图像/音频/表格）输入的处理流程，并加载训练好的模型权重进行推理（`main/SRGA/checkpoints/`）。

## 技术栈
- **后端**：Django（项目入口 `manage.py`，配置位于 `config/`）
- **前端**：Django Templates（HTML 模板位于 `main/templates/`）
- **算法/推理**：PyTorch（模型权重 `.pth` 与 SRGA 代码位于 `main/SRGA/`）
- **开发方式**：Cursor（Vibe Coding）

## 项目结构概览
- `config/`：Django 项目配置（`settings.py`、`urls.py`、`wsgi.py`、`asgi.py`）
- `main/`：主应用（视图、路由、模板等）
  - `templates/`：页面模板
  - `SRGA/`：SRGA 模块（表单、推理逻辑、模型/数据处理代码与配置）
- `scripts/`：常用 PowerShell 脚本（如迁移、运行、环境准备）
- `requirements.txt`：Python 依赖（根目录与 `main/SRGA/` 下可能分别维护）