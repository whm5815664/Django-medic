from django.db import models


class User(models.Model):
    userID = models.CharField(max_length=64, primary_key=True, verbose_name="用户ID")
    userName = models.CharField(max_length=150, verbose_name="姓名")
    userBirth = models.DateField(null=True, blank=True, verbose_name="出生日期")
    userTel = models.CharField(max_length=20, verbose_name="电话")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = "user"
        verbose_name = "用户信息"
        ordering = ["-created_at"]
        managed = False


class Assessment(models.Model):
    userID = models.CharField(max_length=64, db_index=True, verbose_name="用户ID")
    assessment_date = models.DateField(verbose_name="评估日期")
    height = models.FloatField(verbose_name="身高(cm)")
    weight = models.FloatField(verbose_name="体重(kg)")
    bmi = models.FloatField(verbose_name="BMI")
    health_status = models.CharField(max_length=20, verbose_name="健康状况")

    class Meta:
        db_table = "assessments"
        verbose_name = "评估记录"
        ordering = ["-assessment_date"]

    def __str__(self):
        return f"{self.userID} - {self.assessment_date}"
