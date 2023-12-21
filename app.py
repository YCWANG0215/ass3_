import json
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from flask import Flask, request, jsonify, render_template
from azure.cosmos import exceptions, CosmosClient
import time
import math

app = Flask(__name__)


# Redis connection
import redis
# the "Primary connection string" in the "Access keys"
redis_passwd = "KmD13D9JzYSPlkT9yASuREkFGOAWrBnqdAzCaD1MQFU="
redis_host = "wutuo.redis.cache.windows.net"
cache = redis.StrictRedis(
            host=redis_host, port=6380,
            db=0, password=redis_passwd,
            ssl=True,
        )


# 数据库连接部分
DB_CONN_STR = "AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
db_client = CosmosClient.from_connection_string(conn_str = DB_CONN_STR)
database = db_client.get_database_client("tutorial")
container_us_cities = database.get_container_client("us_cities")
container_reviews = database.get_container_client("reviews")

# def calculate_euclidean_distance(city1, city2):
#     return math.sqrt((city1['lat'] - city2['lat']) ** 2 + (city1['lng'] - city2['lng']) ** 2)


def fetch_database(city_name=None, include_header=False, exact_match=False):
    container = database.get_container_client("us_cities")
    QUERY = "SELECT * from us_cities"
    params = None
    if city_name is not None:
        QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name"
        params = [dict(name="@city_name", value=city_name)]
        if not exact_match:
            QUERY = "SELECT * FROM us_cities p WHERE p.city like @city_name"

    headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []
    row_id = 0
    if include_header:
        line = [x for x in headers]
        line.insert(0, "")
        result.append(line)

    for item in container.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        row_id += 1
        line = [str(row_id)]
        for col in headers:
            line.append(item[col])
        result.append(line)
    return result


def fetch_other_database(city_name=None):
    container = database.get_container_client("us_cities")
    QUERY = "SELECT c.city,c.lat, c.lng from us_cities c WHERE c.city != @city_name"
    params = [dict(name="@city_name", value=city_name)]

    # headers = ["city", "lat", "lng", "country", "state", "population"]
    headers = ["city", "lat", "lng"]
    result = []
    # row_id = 0

    for item in container.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        # row_id += 1
        line = []
        for col in headers:
            line.append(item[col])
        result.append(line)
    return result


def get_closest_cities(city_name, page=0, page_size=50):
    start_time = time.time()

    # 有缓存的情况
    cache_key = f"closest_cities:{city_name}:{page}:{page_size}"
    cached_result = cache.get(cache_key)
    if cached_result:
        cached_result = cached_result.decode('utf-8')
        end_time = time.time()
        response = {
            "closest_cities": json.loads(cached_result),
             "compute_time_ms": int((end_time - start_time) * 1000),
             "is_cache": "Yes"
             }
        print(response)
        return jsonify(response)

    # 没有缓存
    print("没有缓存")
    # 获取指定城市的坐标
    city_data = fetch_database(city_name=city_name)
    if not city_data:
        return jsonify({"error": "未找到城市"}), 404

    city_coordinates = {
        "city": (city_data[0][1]),
        "lat": float(city_data[0][2]),
        "lng": float(city_data[0][3])
    }

    filter_cities = fetch_other_database(city_name=city_name)
    print("以获取其他城市信息")

    # 按欧几里得距离排序城市
    # sorted_cities = sorted(filter_cities, key=lambda x: math.sqrt(
    #     (float(x[1]) - city_coordinates.get("lat")) ** 2 + (float(x[2]) - city_coordinates.get("lng")) ** 2))
    calculate_and_add_distance(filter_cities, city_coordinates)
    sorted_cities = sorted(filter_cities, key=lambda x: float(x[-1]))  # Assuming the distance is appended as the last element
    print("计算完距离")
    # 对结果进行分页
    start_index = page * page_size
    end_index = start_index + page_size
    paginated_cities = sorted_cities[start_index:end_index]

    end_time = time.time()
    # 将结果存入缓存
    print("准备存入缓存")
    cache.setex(cache_key, 3600, json.dumps(paginated_cities))
    print("已存入缓存")
    # 准备响应
    response = {
        "closest_cities": paginated_cities,
        "compute_time_ms": int((end_time - start_time) * 1000),
        "is_cache": "No"
    }
    print(response)
    return jsonify(response)

def calculate_and_add_distance(cities, target_coordinates):
    for city in cities[:]:  # Skip the header
        lat_diff = float(city[1]) - target_coordinates["lat"]
        lng_diff = float(city[2]) - target_coordinates["lng"]
        distance = math.sqrt(lat_diff ** 2 + lng_diff ** 2)
        city.append(distance)  # Append the distance to the city data




def euclidean_distance(city1, city2):
    return math.sqrt((city1['lat'] - city2['lat']) ** 2 + (city1['lng'] - city2['lng']) ** 2)


# KNN分类函数
def knn_classify(city, k, seed_cities):
    distances = [(data, euclidean_distance(city, data)) for data in seed_cities]
    distances.sort(key=lambda x: x[1])
    neighbors = [neighbor[0] for neighbor in distances[:k]]
    print(f"neighbors = {neighbors}")
    return neighbors


def calculate_center_city(class_cities):
    # 计算类中所有城市的经度和纬度的平均值
    avg_lat = sum(city['lat'] for city in class_cities) / len(class_cities)
    avg_lng = sum(city['lng'] for city in class_cities) / len(class_cities)

    # 找出离平均经纬度最近的城市作为中心城市
    center_city = min(class_cities, key=lambda city: (city['lat'] - avg_lat) ** 2 + (city['lng'] - avg_lng) ** 2)

    return center_city


@app.route('/data/knn_reviews', methods=['GET'])
def knn_reviews():
    start_time = time.time() * 1000  # Record start time for response time calculation



    # Get request parameters
    classes = int(request.args.get('classes', type=int, default=6))
    k = int(request.args.get('k', type=int, default=3))
    words = int(request.args.get('words', type=int, default=100))

    query = "SELECT us_cities.city, us_cities.lat, us_cities.lng, us_cities.population FROM us_cities"
    # quickly fetch the result if it already in the cache
    # 有缓存
    # 有缓存的情况
    cache_key = f"knn_reviews:{classes}:{k}:{words}"
    cached_result = cache.get(cache_key)
    if cached_result:
        cached_result = cached_result.decode('utf-8')
        end_time = time.time() * 1000
        response = {
            "knn_reviews": json.loads(cached_result),
            "compute_time_ms": int((end_time - start_time)),
            "is_cache": "Yes"
        }
        print(response)
        return jsonify(response)


    # params = [dict(name="@city_name", value='city_name')]
    query_list = list(container_us_cities.query_items(
        query=query,
        enable_cross_partition_query=True
    ))
    for city_info in query_list:
        if city_info['lat'] is not None:
            city_info['lat'] = float(city_info['lat'])
        if city_info['lng'] is not None:
            city_info['lng'] = float(city_info['lng'])
        if city_info['population'] is not None:
            city_info['population'] = int(city_info['population'])

    # for item in query_list:
    #     print(f"city: {item['city']}, lat: {item['lat']}, lng: {item['lng']}")

    # 选一些城市作为初始类别的中心点
    N = classes
    seed_cities = random.sample(query_list, N)
    # print(f"seed_cities = {seed_cities}")
    classified_cities = [[] for _ in range(N)]
    classified_seeds = {}
    classNum = 0
    for index, city in enumerate(seed_cities):
        # classified_cities[seed_cities[0]['city']] = classNum
        # classified_cities[classNum].append(seed_cities[classNum])
        classified_cities[classNum].append(city)
        classified_seeds[city['city']] = classNum
        classNum += 1
    # print(f"classified_cities: {classified_cities}")
    # print(f"classified_seeds: {classified_seeds}")

    for incoming_city in query_list:
        # print(f"incoming_city = {incoming_city}")
        if incoming_city not in seed_cities:
            # 计算当前城市与所有种子城市的欧氏距离
            # distances = [(seed_city, math.sqrt((query_list['lat'] - seed_city['lat'])**2 + (query_list['lng'] - seed_city['lng'])**2)) for seed_city in seed_cities]
            distances = [(seed_city, math.sqrt(
                (query_city['lat'] - seed_city['lat']) ** 2 + (query_city['lng'] - seed_city['lng']) ** 2)) for
                         seed_city in seed_cities for query_city in query_list]
            # 按照距离排序并选择K个最近邻
            distances.sort(key=lambda x: x[1])
            nearest_neighbors = [neighbor[0] for neighbor in distances[:k]]
            # print(f"nearest_neighbors: {nearest_neighbors}")
            # 根据最近邻的类别进行投票，选择最多的类别作为当前城市的类别
            neighbor_classes = [neighbor['city'] for neighbor in nearest_neighbors]
            # print(f"neighbor_classes = {neighbor_classes}")
            # print(f"neighbor_classes[0] = {neighbor_classes[0]}")
            # nearest_city = Counter(neighbor_classes).most_common(1)[0][0]
            # city_class = classified_seeds[neighbor_classes[0]]
            city_class = random.randint(0, N - 1)
            # print(f"city_class = {city_class}")
            # # 把确定的类别city_class赋给city当前城市，并存入一个字典中
            # classified_cities[incoming_city['city']] = city_class
            classified_cities[city_class].append(incoming_city)
            # print(f"classified_cities = {classified_cities}")
            # print()
            # city_class = class
    print(f"classified_cities = {classified_cities}")
    # for index, cities_in_class in enumerate(classified_cities):
    #     print(f"Class {index}:")
    #     for city in cities_in_class:
    #         print(city['city'])
    cities_by_class = {}
    for index, cities_in_class in enumerate(classified_cities):
        cities_by_class[f"Class {index}"] = [city['city'] for city in cities_in_class]

    centers_by_class = {}
    for index, cities_in_class in enumerate(classified_cities):
        center_city = calculate_center_city(cities_in_class)
        centers_by_class[f"Class {index}"] = center_city

    # 构建SQL查询语句，筛选出包含指定城市名的评论
    query2 = "SELECT reviews.city,reviews.score, reviews.review FROM reviews"
    if cache.exists(query2):
        query_list_2 = json.loads(cache.get(query2).decode())
        print("cache hit: [{}]".format(query2))
    else:
        # params = [dict(name="@city_name", value='city_name')]
        query_list_2 = list(container_reviews.query_items(
            query=query2,
            enable_cross_partition_query=True
        ))
        cache.set(query2, json.dumps(query_list_2))
        print("cache miss: [{}]".format(query2))
    print(f"query2 = {query2}")
    for city_info in query_list_2:
        if city_info['score'] is not None:
            city_info['score'] = int(city_info['score'])

    weighted_avg_scores = {}

    for class_index, cities in cities_by_class.items():
        total_weighted_score = 0
        total_weight = 0

    for city_name in cities:
        # Find population and score for the city from query_list and query_list_2
        population = next((item['population'] for item in query_list if item['city'] == city_name), None)
        score = next((item['score'] for item in query_list_2 if item['city'] == city_name), None)

        if population is not None and score is not None:
            # Calculate weighted score
            weighted_score = population * score
            total_weighted_score += weighted_score
            total_weight += population

    # Calculate weighted average score for the class
    if total_weight > 0:
        weighted_avg_scores[class_index] = total_weighted_score / total_weight
    else:
        weighted_avg_scores[class_index] = 0  # or handle the case when total_weight is zero

    # Now you have a dictionary weighted_avg_scores containing weighted average scores for each class



    # # Calculate response time
    end_time = time.time() * 1000

    response_data = {
        'cities_by_class': cities_by_class,
        'center_city': centers_by_class,
        'weighted_avg_score': weighted_avg_scores
    }

    # 没有缓存
    print("准备存入缓存")
    cache.setex(cache_key, 3600, json.dumps(response_data))
    print("已存入缓存")
    # 准备响应
    response = {
        "knn_reviews": response_data,
        "compute_time_ms": int((end_time - start_time)),
        "is_cache": "No"
    }

    return jsonify(response)


# 用于最近城市查询的端点
@app.route('/data/closest_cities', methods=['GET'])
def closest_cities():
    city_name = request.args.get('city')
    page = int(request.args.get('page', 0))
    page_size = int(request.args.get('page_size', 50))

    return get_closest_cities(city_name, page, page_size)


@app.route('/data/knn_reviews', methods=['GET'])
def get_knn_reviews():
    classes = int(request.args.get('classes', 6))
    k = int(request.args.get('k', 3))
    words = int(request.args.get('words', 100))

    start_time = time.time()

    result = knn_reviews(classes, k, words)

    result['time_taken_ms'] = int((time.time() - start_time) * 1000)

    return jsonify(result)


@app.route('/', methods=['GET'])
def words_page():
    return render_template('index.html')


@app.route('/flush_cache', methods=['GET'])
def flush_cache():
    for key in cache.keys():
        cache.delete(key.decode())
    print("delete all cache")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
