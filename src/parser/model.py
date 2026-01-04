def add_model_options(parser):
    group = parser.add_argument_group("Model options")
    group.add_argument("--num_support", type=int, default=3, help="Number of samples in the support set")
    group.add_argument("--num_augment", type=int, default=0, help="Number of augmented samples per support sample")
    group.add_argument("--t_his", type=int, default=30, help="Number of observed frames")
    group.add_argument("--t_pred", type=int, default=30, help="Number of predicted frames")
    group.add_argument("--n_pred", type=int, default=20, help="Number of DCT bases used for prediction")
