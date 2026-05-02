# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation across multiple tasks.

This script runs policy evaluation on multiple registered tasks, supporting various
policy backends (pi0, etc.) with options for subtask tracking and result logging.

Supports multi-env: each "run" spawns num_envs parallel episodes.
Total episodes = num_runs * num_envs.

Usage:
    Run on all registered tasks:
    $ python run_eval.py

    Run on specific tasks:
    $ python run_eval.py --task BananaInBowlTask RubiksCubeTask

    Run on a tag:
    $ python run_eval.py --tag spatial

    Use specific policy:
    $ python run_eval.py --policy pi05

    Run multiple episodes with 2 parallel envs:
    $ python run_eval.py --num-runs 2 --num_envs 4

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output. Default is <timestamp>_<policy>. If you provide the output folder name for a previous run, the script will skip the tasks and episodes that have already been run.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                       help="Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                            "Overrides --remote-host and --remote-port when set.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                       help="Number of actions to execute from each predicted chunk before requesting a new one. "
                            "If omitted, each inference client uses its own default. "
                            "Must match the model's action_horizon for best performance.")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                       help="Which instruction variant to use when a task defines multiple (default, vague, specific, etc.)")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, 'sensor' only, or 'none' (default: all)")
parser.add_argument("--enable-sdf-guidance", "--enable_sdf_guidance", action="store_true",
                    help="Spawn the nvblox sidecar and attach an ESDF voxel grid to each "
                         "policy request as fkc/* keys (required when the openpi server is "
                         "configured with FKC mode != vanilla). Multi-env runs build all "
                         "envs' SDFs in a single batched IPC roundtrip per step.")
parser.add_argument("--nvblox-sidecar-python", "--nvblox_sidecar_python", type=str,
                    default=os.path.join(os.getcwd(), ".venv_nvblox_sidecar/bin/python"),
                    help="Path to the python interpreter that has nvblox_torch installed "
                         "(default: ./.venv_nvblox_sidecar/bin/python).")
parser.add_argument("--sdf-voxel-size", "--sdf_voxel_size", type=float, default=0.025,
                    help="Voxel size in metres for the SDF grid (default: 0.025). Smaller "
                         "= more accurate SDF, more voxels and longer build time.")
parser.add_argument("--sdf-grasp-force-threshold", "--sdf_grasp_force_threshold",
                    type=float, default=0.1,
                    help="Contact force (N) above which a scene object is treated as "
                         "currently grasped and dropped from the obstacle SDF "
                         "(default: 0.1).")
parser.add_argument("--sdf-aabb-padding", "--sdf_aabb_padding", type=float, default=0.0,
                    help="Pad each obstacle's world-axis AABB by this many metres on every "
                         "side. Useful when objects are thinner than one voxel — set to "
                         "~voxel_size for a conservative wrap (default: 0.0).")
parser.add_argument("--sdf-max-esdf-distance", "--sdf_max_esdf_distance",
                    type=float, default=5.0,
                    help="Truncation distance (m) used by nvblox; voxels farther than this "
                         "from any obstacle report this value (default: 5.0).")
parser.add_argument("--report-collisions", "--report_collisions", action="store_true",
                    help="Per-step ground-truth collision logging using IsaacLab body "
                         "positions vs scene-object OBBs. Currently grasped objects are "
                         "automatically excluded. One compressed .npz is written per run.")
parser.add_argument("--collision-body-radius", "--collision_body_radius", type=float, default=0.045,
                    help="Robot-link sphere radius (m) used by the collision reporter to "
                         "decide overlap (default: 0.045 ≈ Franka link half-width).")
parser.add_argument("--collision-obstacle-padding", "--collision_obstacle_padding",
                    type=float, default=0.0,
                    help="Pad each obstacle's world-axis AABB by this many metres on every "
                         "side when checking collisions (default: 0.0).")
parser.add_argument("--collision-grasp-force-threshold", "--collision_grasp_force_threshold",
                    type=float, default=0.1,
                    help="Contact force above which an object is treated as 'grasped' "
                         "and excluded from the collision check (default: 0.1).")
parser.add_argument("--collision-robot-articulation", "--collision_robot_articulation",
                    type=str, default="robot",
                    help="Key under WorldState.articulations whose links count as 'robot' "
                         "for the reporter (default: 'robot').")
parser.add_argument("--collision-body-filter", "--collision_body_filter", nargs="*", default=None,
                    help="Optional list of substrings; only robot bodies whose name contains "
                         "one of these are checked. Useful to skip internal joint stubs.")
# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.eval import create_client, run_episode, summarize_run # noqa
from robolab.core.logging.recorder_manager import patch_recorder_manager # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.utils.print_utils import print_experiment_summary # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results # noqa
import robolab.constants # noqa

# Update robolab.constants module settings from command line arguments
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

# Fix recorder manager
patch_recorder_manager()

# Run automatic factory generation before main
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}"
        if args_cli.instruction_type != "default":
            args_cli.output_folder_name += f"_{args_cli.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
        filter_str = f"tasks: {', '.join(args_cli.task)}"
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
        filter_str = f"tags: {', '.join(args_cli.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=args_cli.policy,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy)

        # Optionally spin up the nvblox sidecar + SDFBuilder for FKC guidance.
        # Done after create_env so the world singleton is populated.
        sdf_builder = None
        if args_cli.enable_sdf_guidance:
            from robolab.core.world.world_state import get_world
            from robolab_policy_client.sdf import SDFBuilder
            from robolab_policy_client.sdf.builder import (
                SDFBuilderConfig,
                WorkspaceGrid,
            )

            world = get_world(env)  # ensure cache is populated
            obstacle_names = tuple(world.objects.keys())
            # Workspace bounds: prefer task-declared if present, else fall
            # back to a Franka-tabletop default (~1m cube in front of base).
            task_cls = type(env_cfg)
            bounds = getattr(task_cls, "sdf_workspace_bounds", None)
            if bounds is None:
                bounds = ((-0.2, -0.6, -0.05), (0.8, 0.6, 1.2))
            workspace = WorkspaceGrid.from_bounds(bounds, args_cli.sdf_voxel_size)
            sdf_cfg = SDFBuilderConfig(
                sidecar_python=args_cli.nvblox_sidecar_python,
                workspace=workspace,
                obstacle_object_names=obstacle_names,
                grasp_force_threshold=args_cli.sdf_grasp_force_threshold,
                aabb_padding=args_cli.sdf_aabb_padding,
                max_esdf_distance_m=args_cli.sdf_max_esdf_distance,
            )
            sdf_builder = SDFBuilder(sdf_cfg).start()
            sdf_builder.set_world(world)
            print(
                f"\033[96m[RoboLab] SDF guidance ON: voxel={workspace.voxel_size}m, "
                f"dims={workspace.grid_dims}, obstacles={obstacle_names}\033[0m"
            )

        # Construct the inference client once per task; reuse across runs.
        # CLI values of None are filtered out by create_client so the
        # client's own defaults apply.
        client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
            remote_uri=args_cli.remote_uri,
            open_loop_horizon=args_cli.open_loop_horizon,
            sdf_builder=sdf_builder,
        )

        # Optional ground-truth collision reporter. Independent of the FKC
        # SDF — uses IsaacLab body positions and ``world.get_bbox``. Cheap
        # per step (<5 ms on a typical scene) and only flushes to disk at
        # episode end.
        collision_reporter = None
        if args_cli.report_collisions:
            from robolab.core.world.world_state import get_world
            from robolab.eval.collision_reporter import (
                CollisionReporter,
                CollisionReporterConfig,
            )

            world_for_reporter = get_world(env)
            obstacle_names_reporter = tuple(world_for_reporter.objects.keys())
            collision_reporter = CollisionReporter(
                world_for_reporter,
                CollisionReporterConfig(
                    obstacle_names=obstacle_names_reporter,
                    robot_articulation_name=args_cli.collision_robot_articulation,
                    body_radius_m=args_cli.collision_body_radius,
                    obstacle_padding_m=args_cli.collision_obstacle_padding,
                    grasp_force_threshold=args_cli.collision_grasp_force_threshold,
                    body_filter=tuple(args_cli.collision_body_filter)
                    if args_cli.collision_body_filter
                    else None,
                ),
            )
            print(
                f"\033[96m[RoboLab] Collision reporting ON: obstacles={obstacle_names_reporter}, "
                f"robot_bodies={len(collision_reporter._body_names)}, "
                f"body_radius={args_cli.collision_body_radius}m\033[0m"
            )

        for run_idx in range(num_runs):

            # Check if all episodes in this run are already complete
            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            # Policy
            if args_cli.instruction_type != "default":
                run_name = task_env + f"_{args_cli.instruction_type}_{run_idx}"
            else:
                run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs)\033[0m")

            collision_npz_path = None
            if collision_reporter is not None:
                collision_npz_path = os.path.join(
                    scene_output_dir, f"collisions_run_{run_idx}.npz"
                )

            env_results, msgs, timing = run_episode(env=env,
                        env_cfg=env_cfg,
                        episode=run_idx,
                        client=client,
                        save_videos=args_cli.save_videos,
                        video_mode=args_cli.video_mode,
                        headless=args_cli.headless,
                        collision_reporter=collision_reporter,
                        collision_output_path=collision_npz_path)

            if collision_reporter is not None:
                summary = collision_reporter.summarize()
                if summary:
                    line = ", ".join(
                        f"env{eid}: {s['steps_in_collision']}/{s['steps_total']} "
                        f"steps ({s['fraction_in_collision']*100:.1f}%), "
                        f"min_clearance={s['min_clearance_m']*1000:.1f}mm"
                        for eid, s in sorted(summary.items())
                    )
                    print(f"\033[96m[RoboLab] Collisions {run_name}: {line}\033[0m")
                    if collision_npz_path:
                        print(f"\033[96m[RoboLab] Collision data → {collision_npz_path}\033[0m")

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                timing=timing,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=args_cli.policy,
                instruction_type=args_cli.instruction_type,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
            )

            # Reset eval state for next run (unfreeze all envs)
            env.reset_eval_state()

        if sdf_builder is not None:
            sdf_builder.close()
        env.close()

    # This will print the results to the terminal, summarized.
    # Alternatively, you can run `python analysis/read_results.py <output_dir>` to read the results from the file.
    summarize_experiment_results(episode_results, show_timing=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
