import json
import random
import requests
import time
import pyfiglet

print("Going to perform 10,000 search queries")
print("#########################")
print("Starting Now")


class RandomQuery:
    def __init__(self):
        with open("queries.json") as f:
            queries = json.load(f)
        self.queries = queries["queries"]
        self.query = None

    def generate_random_query(self):
        query = random.choice(self.queries)
        return query

r = RandomQuery()
response_times = []
total_start = time.perf_counter()
for i in range(10000):
    start = time.perf_counter()
    n = requests.post("http://localhost:9011/search", 
                json={
                    "search_text": r.generate_random_query(),
                    "limit": 100
                    }
                    )
    end = time.perf_counter()
    response_times.append(end - start)
    print(f"Request {i} took {end - start} seconds")
total_end = time.perf_counter()

total_time = total_end - total_start
print(pyfiglet.figlet_format("Total time"))
colorized_output = f"\033[1;32mTotal time: {total_time}\033[0m"
print(colorized_output)
with open("response_times.json", "w") as f:
    json.dump({"response_times": response_times}, f)
