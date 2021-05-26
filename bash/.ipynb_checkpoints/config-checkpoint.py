def get_hparams(dataset):
    print('dataset = ', dataset)
    if dataset == "REST_EN":
        hparams = {
            "SEEDS": "restaurant.english",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,
            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 100,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "REST_SP":
        hparams = {
            "SEEDS": "restaurant.spanish",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,

            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 100,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "REST_FR":
        hparams = {
            "SEEDS": "restaurant.french",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,
            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 100,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "REST_RU":
        hparams = {
            "SEEDS": "restaurant.russian",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,

            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 100,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "REST_DU":
        hparams = {
            "SEEDS": "restaurant.dutch",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,

            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 100,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "REST_TU":
        hparams = {
            "SEEDS": "restaurant.turkish",
            "st_num_aspect": 12,
            "general_asp": 0,
            "aspects": 30,

            "lr": 0.005,
            "dis_mu": 2,
            "hyper_beta": 0.02,
            "gb_temp": 1e-08,
            "w2v_ratio": 0.08,
            "mt_ratio": 0.12,
            "dis_1": 15.0,
            "dis_2": 20.0,
            "dis_3": 50.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "BOOTS":
        hparams = {
            "SEEDS": "boots",
            "st_num_aspect": 9,
            "general_asp": 5,
            "aspects": 30,

            "lr": 0.00000035,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-04,
            "w2v_ratio": 0.1,
            "mt_ratio": 5,
            "dis_1": 16.0,
            "dis_2": 16.0,
            "dis_3": 8.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "BAGS_AND_CASES":
        hparams = {
            "SEEDS": "bags_and_cases",
            "st_num_aspect": 9,
            "general_asp": 4,
            "aspects": 30,

            "lr": 0.00000035,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-03,
            "w2v_ratio": 0.1,
            "mt_ratio": 5,
            "dis_1": 2.0,
            "dis_2": 168.0,
            "dis_3": 168.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "TV":
        hparams = {
            "SEEDS": "tv",
            "st_num_aspect": 9,
            "general_asp": 5,
            "aspects": 30,

            "lr": 0.00000035,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-05,
            "w2v_ratio": 0.1,
            "mt_ratio": 1000,
            "dis_1": 16.0,
            "dis_2": 40.0,
            "dis_3": 64.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "KEYBOARDS":
        hparams = {
            "SEEDS": "keyboards",
            "st_num_aspect": 9,
            "general_asp": 7,
            "aspects": 30,

            "lr": 0.00000015,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-05,
            "w2v_ratio": 0.1,
            "mt_ratio": 3000,
            "dis_1": 128.0,
            "dis_2": 32.0,
            "dis_3": 64.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "VACUUMS":
        hparams = {
            "SEEDS": "vacuums",
            "st_num_aspect": 9,
            "general_asp": 5,
            "aspects": 30,
            "lr": 0.00000015,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-06,
            "w2v_ratio": 0.1,
            "mt_ratio": 4000,
            "dis_1": 128.0,
            "dis_2": 4.0,
            "dis_3": 64.0,
            "aspect_tsne_bt": 20
        }
    elif dataset == "BLUETOOTH":
        hparams = {
            "SEEDS": "bluetooth",
            "st_num_aspect": 9,
            "general_asp": 6,
            "aspects": 30,
            "lr": 0.00000035,
            "dis_mu": 4,
            "hyper_beta": 0.02,
            "gb_temp": 1e-02,
            "w2v_ratio": 0.1,
            "mt_ratio": 2,
            "dis_1": 16.0,
            "dis_2": 168.0,
            "dis_3": 16.0,
            "aspect_tsne_bt": 20
        }
    else:
        raise NameError(f"\'{dataset}\' is not a valid dataset.")

    return hparams