from loggibud.v1.types import CVRPInstance, ParamsVehicles


def createParams(init, finish, cities):
    dirin = "data/cvrp-instances-1.0/dev/"
    dirparam = "data/cvrp-instances-2.0/params/"
    for city in cities:
        for day in range(init,finish):
            paramInstance = "param-"+city.split('-')[1]+'-'+city.split('-')[0]+'-'+str(day)+".json"
            intance = "cvrp-"+city.split('-')[1]+'-'+city.split('-')[0]+'-'+str(day)+".json"
            instance = CVRPInstance.from_file(dirin + city + '/'+intance)
            total_carga = sum(delivery.size for delivery in instance.deliveries)
            min_route_simple = total_carga/instance.vehicle_capacity
            type1 = int(0.4*min_route_simple) # 40% do total de carga
            type2 = int(type1/2) # 20% do total de carga
            type3 = int(type2/2) # 10% do total de carga
            param = ParamsVehicles(
                types=      ["MOTOCYCLE", "VAN", "TRUCK"],
                capacities= [180, 360, 540],
                custs=      [5,10,15],
                num_types=  [type1, type2, type3]
            )
            param.to_file(dirparam + city + '/'+paramInstance)


if __name__ == "__main__":
    cities = ["pa-0","df-0","rj-0"]
    dayStart = 90
    dayFinish = 119
    createParams(dayStart, dayFinish, cities)