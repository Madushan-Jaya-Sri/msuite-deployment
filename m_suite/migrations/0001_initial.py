# Generated by Django 5.0 on 2024-01-16 17:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='keyword_count_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Keyword', models.CharField(max_length=150)),
                ('Count', models.IntegerField()),
                ('Url', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='youtube_comments',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Video_Title', models.CharField(max_length=200)),
                ('Comment', models.TextField()),
            ],
        ),
    ]
