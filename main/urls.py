from django.urls import path

from main import views
from main.Agent import brain_agent, tool

urlpatterns = [
    # 主页
    path("", views.dashboard, name="dashboard"),
    
    # 用户管理页面
    path("api/users/add/", views.add_user_api, name="add_user_api"),
    path("api/users/delete/", views.delete_user_api, name="delete_user_api"),
    
    # 模型
    path("srga/record/", views.srga_record_form, name="srga_record_form"),
    path("srga/reset-temp/", views.srga_reset_temp, name="srga_reset_temp"),
    path("srga/submit/", views.srga_submit, name="srga_submit"),
    path("srga/result/", views.srga_result, name="srga_result"),
    path('analyze/', views.analyze, name='analyze'),

    # 智能体（页面 + 会话 API）
    path("agent/", views.agent_view, name="agent_view"),
    path("agent/session/create", brain_agent.agent_create_session_view, name="agent_session_create"),
    path("agent/session/send", brain_agent.agent_send_message_view, name="agent_session_send"),
    path("agent/session/delete", brain_agent.agent_delete_session_view, name="agent_session_delete"),
    path("agent/session/health-evolution", tool.agent_health_evolution_view, name="agent_health_evolution"),

    # 评估记录页面
    path("user-results/", views.user_results, name="user_results"),
    path("user-results/delete/", views.user_result_delete, name="user_result_delete"),
    
]

