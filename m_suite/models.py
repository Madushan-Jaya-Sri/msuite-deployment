# models.py
from django.db import models

class keyword_count_data(models.Model):
    Keyword = models.CharField(max_length=150)
    Count = models.IntegerField()
    Url = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.Keyword} - {self.Count}"

class youtube_comments(models.Model):
    
    Video_Title = models.CharField(max_length=200)
    Comment = models.TextField()  # Use TextField for potentially longer comments

    def __str__(self):
        return f"{self.Video_Title} - {self.Comment}"


class sentiments_comments(models.Model):
    
    Sentence = models.TextField(max_length=200)
    Sentiment = models.TextField()  # Use TextField for potentially longer comments

    def __str__(self):
        return f"{self.Sentence} - {self.Sentiment}"