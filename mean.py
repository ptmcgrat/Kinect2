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
        return [
            76.5387720376522 / norm_value,  92.73 / norm_value,
            23.425955869324458 / norm_value
        ]
    
    elif dataset == 'patrick':
        return [
            125 / norm_value,  125 / norm_value,
            125 / norm_value
        ]

def get_std(norm_value=255, dataset='activitynet'):
    # Kinetics (10 videos for each class)
    assert dataset in ['activitynet', 'kinetics', 'cichlids']
    if dataset == 'patrick':
        return [
            30 / norm_value, 30 / norm_value, 30 / norm_value
        ]
    
    #return [
    #    38.7568578 / norm_value, 37.88248729 / norm_value,
    #    40.02898126 / norm_value
    #]
