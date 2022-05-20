"""
Creator: Ivanovitch Silva
Date: 18 April 2022
API testing
"""
from fastapi.testclient import TestClient
import os
import sys
import pathlib
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
# for an instance with a low income

def test_get_inference_trap():
    music = {
            "danceability":0.556,
            "energy":0.800,
            "key":"10",
            "loudness":-6.095,
            "mode":"1",
            "speechiness":0.1030,
            "acousticness":0.00713,
            "instrumentalness":0.194000,
            "liveness":0.0897,
            "valence":0.2970,
            "tempo":150.128,
            "duration_ms":213221,
            "time_signature":"4"
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "trap"

def test_get_inference_trap_metal():
    music = {
            "danceability":0.809,
            "energy":0.642,
            "key":"2",
            "loudness":-5.975,
            "mode":"1",
            "speechiness":0.0788,
            "acousticness":0.00214,
            "instrumentalness":0.000250,
            "liveness":0.1350,
            "valence":0.0967,
            "tempo":124.984,
            "duration_ms":154549,
            "time_signature":"4"
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Trap Metal"

def test_get_inference_dark_trap():
    music={
        "danceability": 0.578 ,
        "energy": 0.61 ,
        "key": "7",
        "loudness": -10.375 ,
        "mode": "1",
        "speechiness": 0.0314 ,
        "acousticness": 0.00665 ,
        "instrumentalness": 0.0 ,
        "liveness": 0.177 ,
        "valence": 0.247 ,
        "tempo": 160.099 ,
        "duration_ms": 159702 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Dark Trap"

def test_get_inference_underground_rap():
    music={
        "danceability": 0.701 ,
        "energy": 0.585 ,
        "key": "5",
        "loudness": -7.612999999999999 ,
        "mode": "0",
        "speechiness": 0.132 ,
        "acousticness": 0.344 ,
        "instrumentalness": 0.0 ,
        "liveness": 0.114 ,
        "valence": 0.422 ,
        "tempo": 119.634 ,
        "duration_ms": 216294 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Underground Rap"

def test_get_inference_trance():
    music={
        "danceability": 0.645 ,
       "energy": 0.86 ,
        "key": "6",
        "loudness": -6.4110000000000005 ,
        "mode": "0",
        "speechiness": 0.0853 ,
        "acousticness": 0.00287 ,
        "instrumentalness": 0.805 ,
        "liveness": 0.0676 ,
        "valence": 0.627 ,
        "tempo": 140.013 ,
        "duration_ms": 294857 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "trance"

def test_get_inference_dnb():
    music={
        "danceability": 0.581 ,
        "energy": 0.938 ,
        "key": "7",
        "loudness": -3.63 ,
        "mode": "1",
        "speechiness": 0.0431 ,
        "acousticness": 0.000589 ,
        "instrumentalness": 0.852 ,
        "liveness": 0.107 ,
        "valence": 0.14 ,
        "tempo": 174.017 ,
        "duration_ms": 419410 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "dnb"

def test_get_inference_psytrance():
    music={
    "danceability": 0.725 ,
    "energy": 0.865 ,
    "key": "5",
    "loudness": -7.502000000000002 ,
    "mode": "0",
    "speechiness": 0.0641 ,
    "acousticness": 0.00327 ,
    "instrumentalness": 0.866 ,
    "liveness": 0.0888 ,
    "valence": 0.375 ,
    "tempo": 148.899 ,
    "duration_ms": 320644 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "psytrance"

def test_get_inference_RnB():
    music={
    "danceability": 0.767 ,
    "energy": 0.5760000000000001 ,
    "key": "10",
    "loudness": -9.683 ,
    "mode": "0",
    "speechiness": 0.256 ,
    "acousticness": 0.145 ,
    "instrumentalness": 2.61e-06 ,
    "liveness": 0.0968 ,
    "valence": 0.187 ,
    "tempo": 139.99 ,
    "duration_ms": 96062 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "RnB"

def test_get_inference_hardstyle():
    music={
        "danceability": 0.459 ,
        "energy": 0.904 ,
        "key": "11",
        "loudness": -3.8 ,
        "mode": "1",
        "speechiness": 0.0849 ,
        "acousticness": 0.059 ,
        "instrumentalness": 0.00329 ,
        "liveness": 0.946 ,
        "valence": 0.25 ,
        "tempo": 150.004 ,
        "duration_ms": 250054 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "hardstyle"

def test_get_inference_emo():
    music={
    "danceability": 0.598 ,
    "energy": 0.951 ,
    "key": "1",
    "loudness": -3.7 ,
    "mode": "1",
    "speechiness": 0.138 ,
    "acousticness": 0.0145 ,
    "instrumentalness": 0.0 ,
    "liveness": 0.586 ,
    "valence": 0.638 ,
    "tempo": 137.917 ,
    "duration_ms": 203947 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Emo"

def test_get_inference_rap():
    music={
    "danceability": 0.807 ,
    "energy": 0.544 ,
    "key": "0",
    "loudness": -6.4110000000000005 ,
    "mode": "1",
    "speechiness": 0.33 ,
    "acousticness": 0.1669999999999999 ,
    "instrumentalness": 0.0 ,
    "liveness": 0.2189999999999999 ,
    "valence": 0.455 ,
    "tempo": 144.955 ,
    "duration_ms": 105920 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Rap"

def test_get_inference_techno():
    music={
    "danceability": 0.711 ,
    "energy": 0.926 ,
    "key": "1",
    "loudness": -8.396 ,
    "mode": "0",
    "speechiness": 0.0486 ,
    "acousticness": 0.00116 ,
    "instrumentalness": 0.826 ,
    "liveness": 0.0982 ,
    "valence": 0.314 ,
    "tempo": 128.016 ,
    "duration_ms": 337377 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "techno"

def test_get_inference_hiphop():
    music={
    "danceability": 0.7070000000000001 ,
    "energy": 0.853 ,
    "key": "8",
    "loudness": -5.528 ,
    "mode": "0",
    "speechiness": 0.276 ,
    "acousticness": 0.15 ,
    "instrumentalness": 0.0 ,
    "liveness": 0.765 ,
    "valence": 0.684 ,
    "tempo": 122.027 ,
    "duration_ms": 137365 ,
    "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Hiphop"

def test_get_inference_pop():
    music={
        "danceability": 0.929 ,
        "energy": 0.631 ,
        "key": "1",
        "loudness": -5.989 ,
        "mode": "1",
        "speechiness": 0.104 ,
        "acousticness": 0.152 ,
        "instrumentalness": 0.0 ,
        "liveness": 0.109 ,
        "valence": 0.892 ,
        "tempo": 120.001 ,
        "duration_ms": 144009 ,
        "time_signature": "4",
    }

    r = client.post("/predict", json=music)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "Pop"

