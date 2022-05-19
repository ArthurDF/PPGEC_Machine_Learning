"""
Creator: Ivanovitch Silva
Date: 26 April. 2022
Script that POSTS to the API using the requests 
module and returns both the result of 
model inference and the status code
"""
import requests
import json
# import pprint

music =  {
                "danceability":0.5,
                "energy":0.5,
                "key":"3",
                "loudness":-5,
                "mode":"0",
                "speechiness":0.5,
                "acousticness":0.5,
                "instrumentalness":0.5,
                "liveness":0.5,
                "valence":0.5,
                "tempo":60,
                "duration_ms":300,
                "time_signature":"1"
            }

url = "http://127.0.0.1:8000"
#url = "https://high-income-app.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=music)

print(f"Request: {url}/predict")
print(f"Person: \n danceability: {music['danceability']}\n energy: {music['energy']}\n"\
      f" key: {music['key']}\n loudness: {music['loudness']}\n"\
      f" mode: {music['mode']}\n"\
      f" speechiness: {music['speechiness']}\n"\
      f" acousticness: {music['acousticness']}\n"\
      f" instrumentalness: {music['instrumentalness']}\n"\
      f" liveness: {music['liveness']}\n"\
      f" valence: {music['valence']}\n"\
      f" tempo: {music['tempo']}\n"\
      f" duration_ms: {music['duration_ms']}\n"\
      f" time_signature: {music['time_signature']}\n"
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")