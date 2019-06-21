def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics', 'cichlids']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]
    elif dataset == 'cichlids':
        # Kinetics (10 videos for each class)
        return [
            125 / norm_value, 125 / norm_value,
            125 / norm_value
        ]


def get_std(norm_value=255, dataset ='activitynet'):
    assert dataset in ['activitynet', 'kinetics', 'cichlids']

    # Kinetics (10 videos for each class)
    if dataset == 'cichlids':
        return [
            30 / norm_value, 30 / norm_value,
            30 / norm_value
        ]
