from django.urls import path

from main import views

urlpatterns = [
    # 主页
    path("", views.dashboard, name="dashboard"),
    
    # 用户管理页面
    path("api/users/add/", views.add_user_api, name="add_user_api"),
    
    # 模型
    path("srga/record/", views.srga_record_form, name="srga_record_form"),
    path("srga/reset-temp/", views.srga_reset_temp, name="srga_reset_temp"),
    path("srga/submit/", views.srga_submit, name="srga_submit"),
    path("srga/result/", views.srga_result, name="srga_result"),
    path('analyze/', views.analyze, name='analyze'),

    # 评估记录页面
    path("user-results/", views.user_results, name="user_results"),
    path("user-results/delete/", views.user_result_delete, name="user_result_delete"),
    
]

