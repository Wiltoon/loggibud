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
from loggibud.v1.distances import OSRMConfig

from loggibud.v1.types import (
    Point,
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
)

logger = logging.getLogger(__name__)

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
    vehicles = []
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

def createBatchesPerPackets(deliveries, x: int):
    """Criar lotes pelo numero de pacotes"""
    final_list = lambda deliveries, x: [deliveries[i:i+x] for i in range(0, len(deliveries), x)]
    batchs = final_list(deliveries, x)
    return batchs

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
        pack = order_batch.popleft()
        routePackage(
            vehicles_possibles,
            pack,
            matrix,
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
    return 

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
    # construir o batch crescente/decrescente
    for i in sorted(distances_pack, key = distances_pack.get, reverse=order):
        orderBatchList.append(instance.deliveries[i])
    orderBatch = deque(orderBatchList)
    return 

def meanDistance(p, packetsExist, md):
    d = 0
    for atual in packetsExist:
        if(type(atual) is Point):
            d += md[0][p.idu+1]
        else:
            d += md[p.idu+1][atual.idu+1]
    return d/len(packetsExist)

def routePackage():
    d = 2

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
        instance = CVRPInstance.from_file(file)

        logger.info("Distribuition batch instance.")
        solution = distributionBatch(
            instance, 
            matrix_distance, 
            k_near = args.k_near, 
            batch_size = args.batch_size, 
            organization=args.organization
        )

        logger.info("Starting to dynamic route.")
        for delivery in tqdm(instance.deliveries):
            model_finetuned = route(model_finetuned, delivery)

        solution = finish(instance, model_finetuned)

        solution.to_file(output_dir / f"{instance.name}.json")

    # Run solver on multiprocessing pool.
    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))