import json
from unicodedata import name
from loggibud.v1.types import * 
from collections import defaultdict

def constructVehicles(instance : CVRPInstance, params: ParamsVehicles):
    vehicles = []
    id_v = 1
    for v in range(len(params.types)):
        for _ in range(params.num_types[v]):
            vehicle = Vehicle(
                id = id_v,
                type_vehicle = params.types[v],
                capacity = params.capacities[v],
                cust = params.custs[v],
                origin = instance.origin
            )
            vehicles.append(vehicle)
            id_v += 1
    return vehicles



def instanceToHeterogene(
    instance : CVRPInstance, paramsVehicles : ParamsVehicles):
    name = instance.name
    region = instance.region
    origin = instance.origin
    vehicles = constructVehicles(instance, paramsVehicles)
    deliveries = instance.deliveries
    return CVRPInstanceHeterogeneous(
        name,
        region,
        origin,
        vehicles,
        deliveries
    )

def recreate(dayStart, dayFinish, cities, tip):
    nameDirIn = "data/cvrp-instances-1.0/dev/"
    nameDirOut = "data/cvrp-instances-"+tip+"/dev/"
    nameDirParam = "data/cvrp-instances-"+tip+"/params/"
    for city in cities:
        for day in range(dayStart, dayFinish):
            instanceDir = nameDirIn + city + "/"
            nameInstance = "cvrp-"+city.split('-')[1]+"-"+city.split('-')[0]+"-"+str(day)
            fileDir = instanceDir + nameInstance + ".json"
            instance = CVRPInstance.from_file(fileDir)
            nameParam = "param-"+city.split('-')[1]+"-"+city.split('-')[0]+"-"+str(day)
            paramDir = nameDirParam + city + "/" + nameParam + ".json"
            paramsVehicles = ParamsVehicles.from_file(paramDir)
            instance_heterogeneoun = instanceToHeterogene(instance, paramsVehicles)
            instance_heterogeneoun.to_file(nameDirOut + city + "/"+ nameInstance + ".json")

    return 

if __name__ == "__main__":
    cities = ["pa-0","df-0","rj-0"]
    dayStart = 90
    dayFinish = 120
    tip = "3.0" # tip 2.0 = HETEROGENEO, 3.0 = HOMOGENEO
    recreate(dayStart, dayFinish, cities, tip)