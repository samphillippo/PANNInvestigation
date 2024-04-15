import json
import math
import torch
import numpy as np
import os
from time import time
from DataSet import TrainSampler, EvaluateSampler, collate_fn
from tqdm import tqdm
from sklearn.metrics import average_precision_score

#Mixup class, implements mixup augmentation by generating random coefficients of size batch_size
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    # def get_lambda(self, batch_size):
    #     lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
    #     return np.stack((lam, 1 - lam), axis=1).flatten()


    # returns a list of random coefficients of size batch_size
    def get_lambda(self, batch_size):
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            #uses beta distribution to generate random coefficients
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
num_workers = 8 if (torch.cuda.is_available()) else 1
learning_rate=1e-4
stop_iteration = 2000
holdout_fold = 1
batch_size = 16

#Moves data to device, if it is a float tensor (ignores string numpy arrays)
def move_data_to_device(x):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x).to(device)
    return x

# def compute_mAP(model, data_loader):
#     model.eval()
#     all_targets = []
#     all_predictions = []
#     with torch.no_grad():
#         for batch_data_dict in data_loader:
#             batch_waveform = move_data_to_device(batch_data_dict['waveform'])
#             batch_output = model(batch_waveform)
#             all_predictions.append(batch_output['clipwise_output'].data.cpu().numpy())
#             all_targets.append(batch_data_dict['target'])

#     all_predictions = np.vstack(all_predictions)
#     all_targets = np.vstack(all_targets)

#     average_precisions = []
#     for i in range(all_targets.shape[1]):
#         average_precisions.append(average_precision_score(all_targets[:, i], all_predictions[:, i]))

#     mAP = np.mean(average_precisions)
#     return mAP


#Trains the model on the dataset
def train(model, dataset, workspace, task):
    checkpoints_dir = os.path.join(workspace, 'checkpoints', task)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if device == 'cuda':
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model.to(device)


    #########################BATCH SAMPLING#################################
    train_sampler = TrainSampler(
        data=dataset.data,
        holdout_fold=holdout_fold,
        batch_size=batch_size * 2) #for mixup

    validate_sampler = EvaluateSampler(
        data=dataset.data,
        holdout_fold=holdout_fold,
        batch_size=batch_size)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=validate_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    ########################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    mixup_augmenter = Mixup(mixup_alpha=1.)


    #TODO: IMPORTANT: SAVING

    # print(len(train_loader))
    total_training_samples = len(train_sampler.indexes)
    batch = train_sampler.batch_size
    number_of_batches = math.ceil(total_training_samples / batch)
    print("Total training samples: {}, batch size: {}, number of batches: {}".format(total_training_samples, batch_size, number_of_batches))


    map_values = []  # List to store mAP values for later use
    accs = []  # List to store accuracies for later use
    # Train on mini batches
    iteration = 0
    train_bgn_time = time()
    with open(os.path.join(workspace, 'map_values.json'), 'w') as f:
        for batch_data_dict in tqdm(train_loader):
            #print validation accuracy every 200 iterations
            if iteration % 20 == 0 and iteration > 0 and iteration % 100 != 0:
                print('------------------------------------')
                print('Iteration: {}'.format(iteration))

                train_fin_time = time()

                statistics = evaluate(model, validate_loader)
                print('Validate accuracy: {:.3f}'.format(statistics['accuracy']))

                accs.append(statistics['accuracy'])


                # statistics_container.append(iteration, statistics, 'validate')
                # statistics_container.dump()


                train_time = train_fin_time - train_bgn_time
                validate_time = time() - train_fin_time

                print(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time()


            if (iteration % 100 == 0 and iteration > 0) or iteration == 1:
                train_fin_time = time()
                checkpoint = {
                    'iteration': iteration,
                    'model': model.module.state_dict() if device == 'cuda' else model.state_dict(),}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))

                torch.save(checkpoint, checkpoint_path)
                print('Model saved to {}'.format(checkpoint_path))

                statistics, mAP = evaluate(model, validate_loader)
                print('Validate accuracy: {:.3f}'.format(statistics['accuracy']))

                accs.append(statistics['accuracy'])

                print('Validating model...')
                # mAP = compute_mAP(model, validate_loader)
                print(f'Iteration: {iteration}, mAP: {mAP}')

                # Append mAP to the list
                map_values.append({'iteration': iteration, 'mAP': mAP, 'accuracy': statistics['accuracy']})
                
                # Write mAP to file
                f.write(json.dumps({'iteration': iteration, 'mAP': mAP, 'accuracy': statistics['accuracy']}) + '\n')
                f.flush()

                train_time = train_fin_time - train_bgn_time
                validate_time = time() - train_fin_time

                print(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time()

            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))

            # Move data to GPU
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key])

            # Train
            model.train()

            # print(batch_data_dict['waveform'].shape)
            # print(batch_data_dict['mixup_lambda'].shape)

            #RUNS MODEL WITH MIXUP
            batch_output_dict = model(batch_data_dict['waveform'],
                batch_data_dict['mixup_lambda'])

            mixed_target = (batch_data_dict['target'][0 :: 2].transpose(0, -1) * batch_data_dict['mixup_lambda'][0 :: 2] + \
                batch_data_dict['target'][1 :: 2].transpose(0, -1) * batch_data_dict['mixup_lambda'][1 :: 2]).transpose(0, -1)
            batch_target_dict = {'target': mixed_target}

            loss = - torch.mean(batch_target_dict['target'] * batch_output_dict['clipwise_output'])
            print("iteration: {}, loss: {}".format(iteration, loss))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration == stop_iteration:
                break

            iteration += 1
    # Optionally save the list to a file after training
    with open(os.path.join(workspace, 'map_values_final.json'), 'w') as f:
        json.dump(map_values, f)



###########EVALUATION################

# Helper function to append a value to a dictionary
def append_to_dict(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

#Evaluates the model on the dataset
def evaluate(model, data_loader):
    output_dict = {}
    all_targets = []
    all_predictions = []
    model.eval()

    # Forward data to a model in mini-batches
    with torch.no_grad():
        for n, batch_data_dict in enumerate(data_loader):
            # print(n)
            batch_waveform = move_data_to_device(batch_data_dict['waveform'])

        
            
            batch_output = model(batch_waveform)
            all_predictions.append(batch_output['clipwise_output'].data.cpu().numpy())
            all_targets.append(batch_data_dict['target'])

            append_to_dict(output_dict, 'filename', batch_data_dict['filename'])

            append_to_dict(output_dict, 'clipwise_output',
                batch_output['clipwise_output'].data.cpu().numpy())

            # if return_input:
            #     append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    average_precisions = []
    for i in range(all_targets.shape[1]):
        average_precisions.append(average_precision_score(all_targets[:, i], all_predictions[:, i]))

    mAP = np.mean(average_precisions)

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)


    clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
    target = output_dict['target']    # (audios_num, classes_num)

    #cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)

    #Calculate accuracy
    N = target.shape[0]
    accuracy = np.sum(np.argmax(target, axis=-1) == np.argmax(clipwise_output, axis=-1)) / N

    statistics = {'accuracy': accuracy}

    return statistics, mAP
