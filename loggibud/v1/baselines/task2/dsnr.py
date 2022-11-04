"""
Algorithm Dynamic Search Neighbour Routes: 
Developer: https://www.github.com/Wiltoon
"""
import logging
import os
import numpy as np

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from loggibud.v1.distances import *

from loggibud.v1.types import (
    Point,
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    RoutePossible
)

logger = logging.getLogger(__name__)

class NoRouterFound(Exception):
    def __init__(self, message):
        super().__init__(message)
@dataclass
class DynamicSearchNeighbourRoutes:
    batch_size: int = 75
    k_near: int = 30
    organise: bool = None

def distributionBatch(
        instance: CVRPInstance, 
        matrix_distance, 
        k_near: int, 
        batch_size: int, 
        organization: str
    ) -> CVRPSolution:
    deliveries = []
    vehicles_possibles = {}
    batches = createBatchesPerPackets(instance.deliveries, batch_size)
    for batch in tqdm(batches, desc="DESPATCHING BATCHES"):
        routingBatches(
            instance,
            batch, 
            vehicles_possibles,
            organization,
            matrix_distance,
            k_near,
            deliveries
        )
        solution = buildSolution(instance, vehicles_possibles)
    return solution

def buildSolution(instance: CVRPInstance, vehicles_occupation):
    """Cria a solução e o nome do arquivo json"""
    # dir_out = "out/dinamic/"+city+"/"
    # nameFile = "d"+instance.name+"batch-"+str(NUM_LOTE)+".json"
    # filename = dir_out + nameFile
    name = instance.name
    vehicles = []
    for k, v in vehicles_occupation.items():
        vehicle = []
        dep = 0
        for id_pack in v[0]:
            if dep == 0:
                dep += 1
                continue
            else:
                point = Point(
                    lng=instance.deliveries[id_pack].point.lng, 
                    lat=instance.deliveries[id_pack].point.lat
                )
                delivery = Delivery(
                    id_pack,
                    point,
                    instance.deliveries[id_pack].size,
                    instance.deliveries[id_pack].idu
                )
                vehicle.append(delivery)
        vehicleConstruct = CVRPSolutionVehicle(origin=instance.origin, deliveries=vehicle)
        vehicles.append(vehicleConstruct)
    solution = CVRPSolution(name=name, vehicles=vehicles)
    return solution #, filename

def createBatchesPerPackets(deliveries, x: int):
    """Criar lotes pelo numero de pacotes"""
    final_list = lambda deliveries, x: [deliveries[i:i+x] for i in range(0, len(deliveries), x)]
    batchs = final_list(deliveries, x)
    return batchs


def buildSolution(instance: CVRPInstance, vehicles_occupation):
    """Cria a solução e o nome do arquivo json"""
    name = instance.name
    vehicles = []
    for k, v in vehicles_occupation.items():
        vehicle = []
        dep = 0
        for id_pack in v[0]:
            if dep == 0:
                dep += 1
                continue
            else:
                point = Point(
                    lng=instance.deliveries[id_pack].point.lng, 
                    lat=instance.deliveries[id_pack].point.lat
                )
                delivery = Delivery(
                    id_pack,
                    point,
                    instance.deliveries[id_pack].size,
                    instance.deliveries[id_pack].idu
                )
                vehicle.append(delivery)
        vehicleConstruct = CVRPSolutionVehicle(origin=instance.origin, deliveries=vehicle)
        vehicles.append(vehicleConstruct)
    solution = CVRPSolution(name=name, vehicles=vehicles)
    return solution #, filename

def routingBatches(
            instance,
            batch, 
            vehicles_possibles,
            organization,
            matrix,
            k_near,
            deliveries
    ):
    order_batch = orderBatch(batch, instance, vehicles_possibles, organization, matrix)
    reallocate = []
    while len(order_batch) > 0:
        poss = [p.idu for p in order_batch]
        pack = order_batch.pop(0)
        routePackage(
            vehicles_possibles,
            pack,
            matrix,
            instance,
            order_batch,
            k_near,
            deliveries,
            poss in reallocate # Cancelar a possibilidade de loop
        )
        reallocate.append(poss)
    return order_batch

def orderBatch(batch, instance, vehiclesPossibles, order, md):
    """Se order = 1 é ordem crescente e -1 decrescente"""
    if order != None:
        packetsExist = whoIsOrder(vehiclesPossibles, instance)
        for p in batch:
            packetsExist.append(p)
        orderBatch = buildMetric(instance, batch, packetsExist, order, md)
    else:
        orderBatch = batch
    return orderBatch

def whoIsOrder(vehiclesPossibles, instance):
    deposit = instance.origin
    packets = [deposit]
    for k, v in vehiclesPossibles.items():
        dep = 0
        for id_pack in v[0]:
            if dep == 0:
                dep += 1
            else:
                packets.append(instance.deliveries[id_pack])
    return packets

def buildMetric(instance, batch, packetsExist, order, md):
    distances_pack = {}
    orderBatchList = []
    for p in batch:
        distances_pack[p.idu] = meanDistance(p, packetsExist, md)
    for i in sorted(distances_pack, key = distances_pack.get, reverse=order):
        orderBatchList.append(instance.deliveries[i])
    orderBatch = deque(orderBatchList)
    return orderBatch

def meanDistance(p, packetsExist, md):
    d = 0
    for atual in packetsExist:
        if(type(atual) is Point):
            d += md[0][p.idu+1]
        else:
            d += md[p.idu+1][atual.idu+1]
    return d/len(packetsExist)

def routePackage(
    vPoss, 
    packet: Delivery, 
    md, 
    instance, 
    batch, 
    k_near, 
    deliveries, 
    loop
    ):
    """
    Roteirizando um pacote dinamico (packet)
    """
    attempt = 0
    id_pack = packet.idu
    routes_neig = lookForNeighboringRoutes(packet, deliveries, md, vPoss, k_near) 
    if loop == False:
        for rSelect in routes_neig:
            try:
                insertionPackInRoute(
                    instance, id_pack, vPoss, rSelect, batch, md, deliveries
                ) 
                break
            except NoRouterFound:
                attempt += 1
    if attempt == len(routes_neig) or loop: 
        newRoute, id_route1 = createNewRoute(id_pack, vPoss, 180, deliveries) 
        if attempt != 0:
            rneigh = lookForNeighboringRoutes(packet, deliveries, md, vPoss, 1) 
            r1, r2 = bestLocalTwoOptModificated(
                newRoute, vPoss[rneigh[0]][0], md, instance
            )
            createNewSolution(id_route1, rneigh[0], r1, r2, vPoss)

def insertionPackInRoute(instance: CVRPInstance, id_packet: int, vPoss,
    rSelect, batch_d, md, deliveries):
    """
    Tenta inserir um pacote dinamico na rota, 
    se falhar tenta expulsar o pior pacote da rota selecionada
    """
    route_fake = insertNewPacket(id_packet, vPoss[rSelect][0], md) 
    if capacityRoute(route_fake, instance, vPoss[rSelect][1]): 
        deliveries.append(id_packet)
        vPoss[rSelect][0] = route_fake
    else:
        worst_pack_id = selectWorstPacket(vPoss[rSelect][0], md)
        route_worstpack = createRouteNoWorst(worst_pack_id, vPoss[rSelect][0])
        route_newpacket = insertNewPacket(id_packet, route_worstpack, md)
        if compareRoutes(route_newpacket, vPoss[rSelect][0], md):
            vPoss[rSelect][0] = kickOutWorst(
                instance, vPoss[rSelect][0], worst_pack_id, batch_d, deliveries
            )
            insertionPackInRoute(instance, id_packet, vPoss, rSelect, batch_d, md, deliveries)
        else:
            raise NoRouterFound("Próxima rota!")

def kickOutWorst(instance, route, worst_pack_id, batch_d, deliveries):
    """
    Acrescenta o pior pacote na lista dos não visitados
    e retira o pior pacote da rota original
    """
    batch_d.append(instance.deliveries[worst_pack_id])
    route = createRouteNoWorst(worst_pack_id, route)  
    deliveries.remove(worst_pack_id)
    return route

def compareRoutes(route1, route2, md):
    """
    Compara duas rotas e devolve 
    Verdadeiro se a primeira for melhor e 
    False se a segunda for melhor
    """
    score1 = computeDistanceRoute(route1, md)
    score2 = computeDistanceRoute(route2, md)
    if score1 < score2:
        return True
    else:
        return False

def computeDistanceRoute(route, matrix_distance):
  """Retorna a distancia percorrida pela rota"""
  distance = 0
  for o in range(len(route)-1):
    d = o + 1
    distance += matrix_distance[route[o]+1][route[d]+1]
  return distance

def createRouteNoWorst(id_pack, route):
    """
    Cria uma lista sem o id_pack da rota
    """
    if id_pack == -1:
        return route.copy()
    return [id for id in route if id != id_pack]

def capacityRoute(route, instance: CVRPInstance, capacityMax) -> bool:
    """Retorna verdadeiro se a capacidade da rota foi respeitada"""
    cap = 0
    for id_pack in route:
        cap += instance.deliveries[id_pack].size
    return cap <= capacityMax

def createNewSolution(id_route1, id_route2, r1, r2, vehiclesPossibles):
    """a nova solução deve compor o vehiclePossible"""
    vehiclesPossibles[id_route1] = [r1,vehiclesPossibles[id_route1][1]]
    vehiclesPossibles[id_route2] = [r2,vehiclesPossibles[id_route2][1]]

def insertNewPacket(id_packet, route, matrix_distance):
    """Insere o pacote dinamico na melhor posição da rota"""
    route_supos = route.copy()
    p_insertion = []
    for i in range(1,len(route)+1):
        route_aux = [el for el in route]
        route_aux.insert(i, id_packet)
        p_insertion.append(route_aux)
    scores = []
    for possible in p_insertion:
        score = calculateDiferenceDistanceRoute(matrix_distance, route, possible)
        scores.append(score)
    route_supos = p_insertion[scores.index(min(scores, key = float))] 
    return route_supos    

def calculateDiferenceDistanceRoute(
    matrix_distance, 
    old_possible,
    possible # lista com pacotes de como sera atendido
    ):
  """Calcular a diferença entre uma rota antiga por uma rota nova"""
  distanceOld = 0 
  distanceNew = 0 
  for old in range(len(old_possible)-1):
    dest = old + 1
    distanceOld += matrix_distance[old_possible[old]+1][old_possible[dest]+1]
  for o in range(len(possible)-1):
    d = o + 1
    distanceNew += matrix_distance[possible[o]+1][possible[d]+1]
  distance = distanceNew - distanceOld
  return distance

def availablePack(route, i, md):
    """
    Avaliar o score de um pacote dentro de uma rota
    """
    if i >= 1: # a rota tem q ter pelo menos um pacote
        id_prev = route[i-1]+1
        id_act = route[i]+1
        if i == (len(route)-1): # ultimo elemento
            # id do prox pack = route[i-1] | route[i-1]+1 ?
            return md[id_prev][id_act]
        else:
            id_next = route[i+1]
            return md[id_prev][id_act] + md[id_act][id_next] - md[id_prev][id_next]
    else:
        return 0 

def selectWorstPacket(route, md):
    """
    Selecionar o id do pior pacote da rota CONSIDERE O DEPOSITO
    """
    worstScore = 0
    id_worst = -1
    if len(route) > 0:
        for i in range(1,len(route)):
            score = availablePack(route, i, md)
            if worstScore < score:
                id_worst = route[i]
    return id_worst

def bestLocalTwoOptModificated(route1, route2, md, instance):
    """Ajusta as melhores posições para as rotas"""
    routes = [route1, route2]
    bestScore = calculateDistanceRoutes(routes, md)+1
    score = bestScore-1
    while score < bestScore:
        bestScore = score
        score = twoOptStarModificatedScore(route1, route2, md, instance) 
    return route1, route2

def twoOptStarModificatedScore(route1, route2, md, instance):
    p1, p2 = 1, 1
    routes = [route1, route2]
    while p1 < len(route1) and p2 < len(route2):
        # print(p1,p2)
        possibles_solutions = constructPossiblesSolutions(
            md, instance, 
            route1, route2, 
            p1, p2)
        best_solution = selectPossible(possibles_solutions)
        if best_solution != None:
            route1, route2  = best_solution.route1.copy(), best_solution.route2.copy()
            if best_solution.select[0] != best_solution.select[1]:
                p1 += 1
                p2 += 1
            else:
                if best_solution.select[0] == 1:
                    p1 += 1
                else:
                    p2 += 1
        else:
            p1 += 1
            p2 += 1
    return calculateDistanceRoutes(routes, md)

def selectPossible(possibles: List[RoutePossible]):
    selected = None
    value_s = 0
    for possible in possibles:
        if possible.gain > value_s:
            selected = possible
            value_s = possible.gain
    return selected

def compensationRemoveInRoute(
    md, 
    ro, 
    i_remove # index -> ro[i_remove] = 152 -> 153 
):
    lose = 0 # ... calcular as arestas perdidas
    gain = 0 # ... calcular as arestas adicionadas
    if i_remove < (len(ro)-1): # ultimo elemento não tem perda
        if i_remove == 1:
            gain += md[ro[i_remove-1]][ro[i_remove+1]+1]
            lose += md[ro[i_remove-1]][ro[i_remove]+1]
            lose += md[ro[i_remove]+1][ro[i_remove+1]+1]
        else:
            gain += md[ro[i_remove-1]+1][ro[i_remove+1]+1]
            lose += md[ro[i_remove-1]+1][ro[i_remove]+1]
            lose += md[ro[i_remove]+1][ro[i_remove+1]+1]
    else:
        if i_remove == 1:
            lose += md[ro[i_remove-1]][ro[i_remove]+1]
        else:
            lose += md[ro[i_remove-1]+1][ro[i_remove]+1]
    return round((lose - gain)/1_000,3)

def compensationInsertInRoute(
    md,
    rd,
    p_insert, # 152 -> 153 
    i_destiny # index = 3 
    ):
    gainDistanceTotal = 0
    lose = 0
    gain = 0
    if i_destiny < (len(rd)-1): # ultimo elemento não tem perda
        if i_destiny == 1:
            lose += md[rd[i_destiny-1]][rd[i_destiny]+1]
            gain += md[rd[i_destiny-1]][p_insert+1]
            gain += md[p_insert+1][rd[i_destiny]+1]
        else:
            lose += md[rd[i_destiny-1]+1][rd[i_destiny]+1]
            gain += md[rd[i_destiny-1]+1][p_insert+1]
            gain += md[p_insert+1][rd[i_destiny]+1]
    else:
        if i_destiny == 1:
            gain += md[rd[i_destiny-1]][p_insert+1]
        else:
            gain += md[rd[i_destiny-1]+1][p_insert+1]
    gainDistanceTotal = round((lose - gain)/1_000,3)
    return gainDistanceTotal

def insertBothInRoute(
    md, instance,
    r1, r2, 
    i_insert, selected: int):
    removePack = compensationRemoveInRoute(md, r2, i_insert)
    possibles = []
    select = [selected, selected]
    if constraintCapacityAdd(instance, r1, instance.deliveries[r2[i_insert]]):
        for i in range(1,len(r1)+1):
            poss = {}
            new_route1 = r1.copy()
            new_route1.insert(i, r2[i_insert])
            gain = compensationInsertInRoute(md, r1, r2[i_insert], i)
            new_route2 = [cor for cor in r2 if cor != r2[i_insert]] # remover o r2[p_insert]
            if gain > 0:
                poss = RoutePossible(new_route1, new_route2, round((gain+removePack),3), select)
                possibles.append(poss)
        possibleInsert = selectPossible(possibles)
    else:
        return None # nao ocupou espaço
    return possibleInsert

def constraintCapacityAdd(instance, route, arrive: Delivery):
    CAP = instance.vehicle_capacity
    DEP = 0
    total = 0 
    for i in route:
        if DEP == 0:
            DEP += 1
        else:
            total += instance.deliveries[i].size
    packSize = arrive.size
    return total + packSize <= CAP

def constructPossiblesSolutions(
    md, instance: CVRPInstance, 
    route1: list, route2: list, 
    p1: int, p2: int):
    possibles = []
    swap = swapPacks(md, instance, route1, route2, p1, p2)
    if swap != None:
        possibles.append(swap)
    possibleRoute1 = insertBothInRoute(md, instance, route1, route2, p2, 1) # p2 -> route1
    if possibleRoute1 != None:
        possibles.append(possibleRoute1)
    possibleRoute2 = insertBothInRoute(md, instance, route2, route1, p1, 2) # p1 -> route2
    if possibleRoute2 != None:
        possibles.append(possibleRoute2)
    return possibles

def calculateDistanceRoutes(routes, matrix_distance):
  """Calcular a distancia percorrida por uma lista de rota"""
  distance = []
  for route in routes:
    distance.append(computeDistanceRoute(route, matrix_distance))
  return sum(distance)

def swapPossible(instance, route1, route2, p1, p2):
    # somar todas as cargas da rota1 - packet1 + packet2
    return constraintCapacity(
        instance, 
        route1, p1, instance.deliveries[route2[p2]]) and constraintCapacity(
            instance, 
            route2, p2, instance.deliveries[route1[p1]])

def constraintCapacity(instance, route, left, arrive: Delivery):
    CAP = instance.vehicle_capacity
    DEP = 0
    total = 0 
    for i in route:
        if DEP == 0:
            DEP += 1
        else:
            total += instance.deliveries[i].size
    sizeLeft = instance.deliveries[route[left]].size
    packSize = arrive.size
    return total - sizeLeft + packSize <= CAP

def compensationSwap(md, r1, r2, i, j):
    if i != (len(r1)-1):
        if i-1 == 0:
            gain = md[r1[i-1]][r2[j]+1] + md[r2[j]+1][r1[i+1]+1]
            lose = md[r1[i-1]][r1[i]+1] + md[r1[i]+1][r1[i+1]+1]
        else:
            gain = md[r1[i-1]+1][r2[j]+1] + md[r2[j]+1][r1[i+1]+1]
            lose = md[r1[i-1]+1][r1[i]+1] + md[r1[i]+1][r1[i+1]+1]
    else:
        if i-1 == 0:
            gain = md[r1[i-1]][r2[j]+1]
            lose = md[r1[i-1]][r1[i]+1]
        else:
            gain = md[r1[i-1]+1][r2[j]+1]
            lose = md[r1[i-1]+1][r1[i]+1]
    return gain, lose

def computeCompensationSwap(md, route1, route2, p1, p2):
    gain1, lose1 = compensationSwap(md, route1, route2, p1, p2)
    gain2, lose2 = compensationSwap(md, route2, route1, p2, p1)
    loseSwap = lose1 + lose2
    gainSwap = gain1 + gain2
    return round((gainSwap)/1_000,3), round((loseSwap)/1_000, 3)

def swapPacks(
    matrix_distance, 
    instance: CVRPInstance, 
    route1: list, route2: list, 
    p1: int, p2: int) -> RoutePossible:
    if swapPossible(instance, route1, route2, p1, p2):
        select = [2, 1]
        gainSwap, loseSwap = computeCompensationSwap(
                    matrix_distance, route1, route2, p1, p2)
        if loseSwap > gainSwap:
            new_route1 = [
                route1[i] 
                for i in range(len(route1)) 
                if i != p1
            ]
            new_route1.insert(p1, route2[p2])
            new_route2 = [
                route2[j] 
                for j in range(len(route2)) 
                if j != p2
            ]
            new_route2.insert(p2, route1[p1])
            gain = round((loseSwap - gainSwap), 3)
            possibleSwap = RoutePossible(
                new_route1,
                new_route2,
                gain,
                select)
        else:
            return None # lose < gain
    else:
        return None # nao respeita capacidade
    return possibleSwap

def createNewRoute(packet_id, vehiclesPossibles, newCap, deliveries):
    """Criar uma nova rota do deposito até o packet"""
    try:
        newVehicle = max(vehiclesPossibles) + 1
    except ValueError:
        newVehicle = 1
    vehiclesPossibles[newVehicle] = [[0, packet_id], newCap]
    deliveries.append(packet_id)
    return vehiclesPossibles[newVehicle][0], newVehicle

def lookForNeighboringRoutes(packet: Delivery, deliveries, md, vPoss: dict, T: int):
    """Procura pelas rotas vizinhas"""
    routes_neigs = []
    neighs = {}
    packs_neigs = []
    for i in deliveries:
      neighs[i] = md[packet.idu+1][i+1]
    if len(deliveries) != 0:
        for id in sorted(neighs, key = neighs.get):
            packs_neigs.append(id)
        auxT = 0
        for d in packs_neigs:
            if auxT < T:
                for k, v in vPoss.items():
                    if d in v:
                        if k not in routes_neigs:
                            routes_neigs.append(k)
                        auxT += 1
    return routes_neigs

def solve_instance(
    instance: CVRPInstance
) -> CVRPSolution:
    """Solve an instance distribution packages of batch"""
    return CVRPSolution(instance)

if __name__ == "__main__":
    osrm_config = OSRMConfig(
        host="http://ec2-34-222-175-250.us-west-2.compute.amazonaws.com"
    )
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--k_near", type=str, required=True)
    parser.add_argument("--eval_instances", type=str, required=True)
    parser.add_argument("--organization", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--params", type=str)
    args = parser.parse_args()
    args.organization = None
    # Load instance and heuristic params.
    eval_path = Path(args.eval_instances)
    eval_path_dir = eval_path if eval_path.is_dir() else eval_path.parent
    eval_files = (
        [eval_path] if eval_path.is_file() else list(eval_path.iterdir())
    )

    params = None

    output_dir = Path(args.output or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    def solve(file):
        osrm_config = OSRMConfig(
            host="http://ec2-34-222-175-250.us-west-2.compute.amazonaws.com"
        )
        instance = CVRPInstance.from_file(file)
        for i in range(len(instance.deliveries)):
            instance.deliveries[i].idu = i
        origin = [instance.origin]
        deliveries = [d.point for d in instance.deliveries]
        points = [*origin, *deliveries]
        matrix_distance = calculate_distance_matrix_m(
            points, osrm_config
        )
        logger.info("Distribuition batch instance.")
        solution = distributionBatch(
            instance, 
            matrix_distance, 
            k_near = int(args.k_near), 
            batch_size = int(args.batch_size), 
            organization=args.organization
        )

        # logger.info("Starting to dynamic route.")
        # for delivery in tqdm(instance.deliveries):
        #     model_finetuned = route(model_finetuned, delivery)

        # solution = finish(instance, model_finetuned)

        solution.to_file(output_dir / f"{instance.name}.json")
    print(eval_path)
    solve(eval_path)
    # Run solver on multiprocessing pool.
    # with Pool(os.cpu_count()) as pool:
    #     list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))