"""
Algorithm Dynamic Search Neighbour Routes: 
Developer: https://www.github.com/Wiltoon
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
from loggibud.v1.distances import OSRMConfig

from loggibud.v1.types import (
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
)

logger = logging.getLogger(__name__)

@dataclass
class DynamicSearchNeighbourRoutes:
    num_packages_per_batch: int = 75
    t_closest_packages: int = 30

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