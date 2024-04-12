import torch

num_workers = 8 #follows format from given code. Change?

def train(model, dataset):
    if torch.cuda.is_available():
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    #TODO use sampler??

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=validate_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)


    #ISSUE: do we need to store test data???
    #They just use validate!!! (i guess this is ok cause we're technically tuning?)


    #COLLATE_FN
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...},
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
