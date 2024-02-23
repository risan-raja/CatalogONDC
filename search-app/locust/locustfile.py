import json
import random

from locust import FastHttpUser, task, between



class RandomQuery:
    def __init__(self):
        with open("queries.json") as f:
            queries = json.load(f)
        self.queries = queries["queries"]
        self.query = None

    def __call__(self):
        query = random.choice(self.queries)
        return query

class SearchUser(FastHttpUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_query = RandomQuery()

    wait_time = between(5, 9)
    @task
    def hello_world(self):
        self.client.post("http://localhost:9011/search", 
                         json={
                             "search_text": self.random_query(),
                              "limit": 10
                              })


