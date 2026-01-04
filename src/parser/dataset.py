from src.datasets.dataset import POSE_REPS


def add_dataset_options(parser):
    group = parser.add_argument_group("Dataset options")
    group.add_argument("--dataset", type=str, default="humanact12", help="Dataset to load")
    group.add_argument("--num_frames", type=int, default=60, help="Number of frames or -1 => whole, -2 => random between min_len and total")
    group.add_argument("--num_joints", type=int, default=24, help="Number of joints")
    group.add_argument("--coord_dim", type=int, default=3, help="Coordinate dimension per joint (e.g., 2 for 2D, 3 for 3D)")
    group.add_argument("--pose_rep", type=str, default="xyz", choices=POSE_REPS, help="xyz or rotvec etc")

    group.add_argument("--glob", dest="glob", action="store_true", help="If we want global rotation")
    group.add_argument("--no-glob", dest="glob", action="store_false", help="If we don't want global rotation")
    group.set_defaults(glob=True)
    group.add_argument("--glob_rot", type=int, nargs="+", default=[3.141592653589793, 0, 0],
                       help="Default rotation, usefull if glob is False")
    
    group.add_argument("--translation", dest="translation", action="store_true",
                       help="If we want to output translation")
    group.add_argument("--no-translation", dest="translation", action="store_false",
                       help="If we don't want to output translation")
    group.set_defaults(translation=True)
