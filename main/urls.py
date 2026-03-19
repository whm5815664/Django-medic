from django.urls import path

from main import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("srga/record/", views.srga_record_form, name="srga_record_form"),
    path("srga/reset-temp/", views.srga_reset_temp, name="srga_reset_temp"),
    path("srga/submit/", views.srga_submit, name="srga_submit"),
    path("srga/result/", views.srga_result, name="srga_result"),
]

