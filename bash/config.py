def get_hparams(dataset):
    print('dataset = ', dataset)
    if dataset == "cell_core":
        hparams = {
            "model": "UCPR",
            "gp_setting": "6_800_15_500_50",
            "lr": 1e-04,
            "lambda_num": 2.0,
            "n_memory": 64,
            "p_hop": 2,
            "reasoning_step": 2,
            "embed_size": 32
        }
    elif dataset == "beauty_core":
        hparams = {
            "model": "UCPR",
            "gp_setting": "6_800_15_500_50",
            "lr": 1e-04,
            "lambda_num": 0.5,
            "n_memory": 64,
            "p_hop": 2,
            "reasoning_step": 2,
            "embed_size": 32
        }
    elif dataset == "cloth_core":
        hparams = {
            "model": "UCPR",
            "gp_setting": "6_800_15_500_50",
            "lr": 1e-04,
            "lambda_num": 0.3,
            "n_memory": 32,
            "p_hop": 2,
            "reasoning_step": 2,
            "embed_size": 32
        }
    elif dataset == "MovieLens-1M_core":
        hparams = {
            "model": "UCPR",
            "gp_setting": "6000_800_15_500_50",
            "lr": 1e-04,
            "lambda_num": 2.0,
            "n_memory": 64,
            "p_hop": 2,
            "reasoning_step": 4,
            "embed_size": 32
        }
    elif dataset == "amazon-book_20core":
        hparams = {
            "model": "UCPR",
            "gp_setting": "6000_800_15_500_50",
            "lr": 1e-04,
            "lambda_num": 0.2,
            "n_memory": 64,
            "p_hop": 2,
            "reasoning_step": 3,
            "embed_size": 32
        }
    else:
        raise NameError(f"\'{dataset}\' is not a valid dataset.")

    return hparams