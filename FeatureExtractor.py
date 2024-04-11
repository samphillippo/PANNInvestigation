

def extract_features(dataset):
    """
    Extract features from the dataset.
    :param dataset: The dataset to extract features from.
    :return: The features extracted from the dataset.
    """
    features = []
    for data in dataset:
        feature = []
        feature.append(data['feature1'])
        feature.append(data['feature2'])
        feature.append(data['feature3'])
        feature.append(data['feature4'])
        features.append(feature)
    return features
