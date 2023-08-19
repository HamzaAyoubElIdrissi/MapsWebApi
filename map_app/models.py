from django.db import models



class Address(models.Model):
    id = models.AutoField(primary_key=True)
    address = models.CharField(max_length=255)
    latitude = models.DecimalField(max_digits=20, decimal_places=15)
    longitude = models.DecimalField(max_digits=20, decimal_places=15)
    cluster = models.IntegerField()

    def __str__(self):
        return self.address