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

def distributionBatch():
    d = 1

def routePackage():
    d = 2

def solve_instance(
    instance: CVRPInstance
) -> CVRPSolution:
    """Solve an instance distribution packages of batch"""
    return CVRPSolution(instance)

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--eval_instances", type=str, required=True)
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
        model_finetuned = distributionBatch(model, instance)

        logger.info("Starting to dynamic route.")
        for delivery in tqdm(instance.deliveries):
            model_finetuned = route(model_finetuned, delivery)

        solution = finish(instance, model_finetuned)

        solution.to_file(output_dir / f"{instance.name}.json")

    # Run solver on multiprocessing pool.
    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))