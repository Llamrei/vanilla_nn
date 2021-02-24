def manhattan(a: dict, b: dict):
    dist = 0
    for key in a:
        dist += abs(a[key]-b[key])
    return dist
