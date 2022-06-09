from loggibud.v1.types import CVRPInstance, ParamsVehicles


def createParams(init, finish, cities):
    dirout = "cvrp-instances-1.0/dev/"
    dirparam = "cvrp-instances-2.0/params/"
    for city in cities:
        for day in range(init,finish):
            paramInstance = "param-"+city.split('-')[1]+'-'+city.split('-')[0]+'-'+str(day)+".json"
            intance = "cvrp-"+city.split('-')[1]+'-'+city.split('-')[0]+'-'+str(day)+".json"
            instance = CVRPInstance.from_file(dirout + city + '/'+intance)
            type1 = 4
            type2 = 2
            type3 = 1
            param = ParamsVehicles(
                types=["MOTOCYCLE", "VAN", "TRUCK"]
                capacities=["180", "360", "540"]
                custs=[5,10,15]
                num_types=[type1, type2, type3]
            )


if __name__ == "__main__":
    cities = ["pa-0","df-0","rj-0"]
    dayStart = 90
    dayFinish = 119
    createParams(dayStart, dayFinish, cities)