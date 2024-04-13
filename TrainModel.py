import torch
import numpy as np

#Mixup class, implements mixup augmentation by generating random coefficients of size batch_size
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            #uses beta distribution to generate random coefficients
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


num_workers = 8 #follows format from given code. Change?
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
learning_rate=1e-4

def train(model, dataset):
    if device == 'cuda':
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model.to(device)


    ############################BATCHING######################################
    #TODO use sampler??

    #sampler is supposed to get batch_size individual samples,
    #TODO: how

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

    ########################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    mixup_augmenter = Mixup(mixup_alpha=1.)


# def evaluate(model, data_loader):
#     output_dict = forward(
#         model=self.model,
#         generator=data_loader,
#         return_target=True)

#     clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
#     target = output_dict['target']    # (audios_num, classes_num)

#     cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
#     accuracy = calculate_accuracy(target, clipwise_output)

#     statistics = {'accuracy': accuracy}

#     return statistics
