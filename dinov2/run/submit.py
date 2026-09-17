# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import submitit

from dinov2.utils.cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_user_checkpoint_path,
)


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    slurm_partition = get_slurm_partition()
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--ngpus",
        "--gpus",
        "--gpus-per-node",
        default=1,
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--nodes",
        "--nnodes",
        default=1,
        type=int,
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--timeout",
        default=2800,
        type=int,
        help="Duration of the job",
    )
    parser.add_argument(
        "--partition",
        default=slurm_partition,
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--use-volta32",
        action="store_true",
        help="Request V100-32GB GPUs",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Nodes to exclude",
    )
    return parser


def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = user_checkpoint_path / "experiments"
    path.mkdir(exist_ok=True)
    return path


def submit_jobs(task_class, args, name: str):
    if not args.output_dir:
        args.output_dir = str(get_shared_folder() / "%j")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #submitit_folder = Path("/tmp") / f"dinov2_submitit_{os.environ.get('USER', 'user')}"
    
    submitit_folder = (
        Path("/mnt/users")
        / os.environ.get("USER", "user")
        / "dinov2"
        / "submitit"
    )

    submitit_folder.mkdir(parents=True, exist_ok=True)

    #executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    executor = submitit.AutoExecutor(
        folder=str(submitit_folder),
        slurm_max_num_timeout=30,
    )

    kwargs = {}
    if args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.exclude:
        kwargs["slurm_exclude"] = args.exclude

    executor_params = get_slurm_executor_parameters(
        nodes=args.nodes,
        num_gpus_per_node=args.ngpus,
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition="high_performance",
        **kwargs,
    )
    
    outer_job_id = os.environ.get("SLURM_JOB_ID")

    if outer_job_id:
        executor_params["slurm_dependency"] = f"afterok:{outer_job_id}"

    env_dir = "/local/$USER/env_dinov2"
    repo_dir = "/mnt/users/$USER/dinov2"

    executor.update_parameters(
            name=name, 
            **executor_params,
             local_setup=[
                    "module purge",
                    "module load python/3.10.13",

                    # Cria o ambiente caso ele ainda não exista
                    f"""
                    if [ ! -x {env_dir}/bin/python ]; then
                        echo "=============================================="
                        echo "Ambiente DINOv2 não encontrado."
                        echo "Criando: {env_dir}"
                        echo "=============================================="

                        python -m venv {env_dir}

                        {env_dir}/bin/python -m pip install --upgrade pip

                        {env_dir}/bin/python -m pip install -r {repo_dir}/requirements.txt 

                        echo "=============================================="
                        echo "Ambiente criado com sucesso."
                        echo "=============================================="
                    else
                        echo "=============================================="
                        echo "Ambiente DINOv2 encontrado:"
                        echo "{env_dir}"
                        echo "=============================================="
                    fi
                    """,

                    # Usa o ambiente criado
                    f"export PATH={env_dir}/bin:$PATH",
                    f"export VIRTUAL_ENV={env_dir}",

                    # Código do DINOv2
                    "unset PYTHONPATH",
                    f"export PYTHONPATH={env_dir}/lib/python3.10/site-packages:{repo_dir}",

                    f"""
                    echo "=============================================="
                    echo "TESTE DO AMBIENTE"
                    echo "=============================================="
                    which python
                    python -c "import torch; print('PyTorch:', torch.__version__); print('Torch path:', torch.__file__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    """,
                    
                ],
    )
    

    task = task_class(args)
    job = executor.submit(task)

    logger.info(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(args.output_dir).replace("%j", str(job.job_id))
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")
