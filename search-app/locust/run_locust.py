import json
import random

from locust import HttpUser, task, between

with open("queries.json") as f:
    queries = json.load(f)

class SearchUser(HttpUser):

    @task
    def hello_world(self):
        self.client.get("/hello")
