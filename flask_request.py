import random
import requests

url = "http://localhost:5000/predict"

feat_list = []
for _ in range(784):
	feat_list.append(random.random())
req = { "feature_array" : feat_list}

response = requests.post(url, json=req)
print(response.status_code)
# print(response.json())