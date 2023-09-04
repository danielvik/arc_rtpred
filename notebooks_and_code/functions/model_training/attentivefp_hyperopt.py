import deepchem as dc
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from hyperopt import hp, fmin, tpe, Trials, rand, plotting
from matplotlib import pyplot as plt

seed = 42

############################################################################
# Parameters
############################################################################

parser = argparse.ArgumentParser()

#input/output
parser.add_argument('--train_dir', type=str, default='../data/premade_splits/molgraphconv/cv_splits/train_0/', help='directory for featurized data -- defaults to split 0')
parser.add_argument('--val_dir', type=str, default='../data/premade_splits/molgraphconv/cv_splits/valid_0/', help='directory for featurized data -- defaults to split 0')
parser.add_argument('--test_dir', type=str, default='../data/premade_splits/molgraphconv/test_data/', help='directory for featurized data')


parser.add_argument('--model_directory', type=str, default='../models/model_default/', help='directory for model_saving')

# training
parser.add_argument('--batchsize', type=int, default=35, help='batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--callback_intervals', type=int, default=1000, help='Number of epochs without change for callback to trigger')


# hyperparameter optimization 
parser.add_argument('--iterations', type=int, default=20, help='number of hyperparameter optimization iterations')



args = parser.parse_args()


print('\n** This is the AttentiveFP Regressor, for retention-time prediction, RMSE optimized **\n')

############################################################################
# Reading featurized, and split data
############################################################################

train_dataset = dc.data.DiskDataset(data_dir = args.train_dir)
valid_dataset = dc.data.DiskDataset(data_dir = args.val_dir)
test_dataset = dc.data.DiskDataset(data_dir = args.test_dir)

print('*** reloaded the featurized data ***')
print(train_dataset)

############################################################################
# Defining Search Space
############################################################################

search_space = {
     ##* AttentiveFP
     'num_layers' : hp.randint('num_layers', high = 6, low = 1), # deepchem only have one layer parameter, so assuming this is the combined from atom and molecule layers
     'graph_feat_size' : hp.randint('graph_feat_size', high = 300, low = 30), # assuming this is the fingerprint dimesions
     'dropout': hp.uniform('dropout',high=0.5, low=0.0),
     'learning_rate' : hp.loguniform('learning_rate', high = (np.log(0.1)), low = (np.log(0.0001))),
     'weight_decay_penalty' : hp.loguniform('weight_decay_penalty', high = (np.log(0.01)), low = (np.log(0.00001)))}

############################################################################
# Model training
############################################################################

model_check_point = (args.model_directory)#'../models/deepchem/dmpnn_rdkit/hyperopt3/'
os.makedirs(os.path.dirname(model_check_point), exist_ok=True)
print('saving model to: ', model_check_point)

metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
epochs = args.epochs
batch_size = args.batchsize
validation_interval = args.callback_intervals
n_tasks = len(train_dataset.tasks)

def fm(search_args):
    model = dc.models.AttentiveFPModel(n_tasks = n_tasks,
                                    mode = 'regression',
                                    model_dir = model_check_point,
                                    batch_size = batch_size,
                                    
                                    ##* AttentiveFP Search Arguments
                                    num_layers = search_args['num_layers'],
                                    graph_feat_size = search_args['graph_feat_size'],
                                    dropout = search_args['dropout'],
                                    learning_rate = search_args['learning_rate'],
                                    weight_decay = search_args['weight_decay_penalty']
                                    )
                                        
    #validation callback that saves the best checkpoint, i.e the one with the maximum score.
    
    callbacks = dc.models.ValidationCallback(valid_dataset, 
                                            validation_interval, 
                                            [metric],
                                            save_dir=model_check_point,
                                            #transformers=transformers,
                                            save_on_minimum=True)
    
    model.fit(train_dataset, nb_epoch = epochs, callbacks = callbacks)

    #restoring the best checkpoint and passing the negative of its validation score to be minimized.
    model.restore(model_dir=model_check_point)
    valid_score = model.evaluate(valid_dataset, [metric])#, transformers)
    #print('validation score:',valid_score['mean_squared_error'])
    return valid_score['mean_squared_error']

############################################################################
# Hyperparameter optimization trials
############################################################################

trials=Trials()
best = fmin(fm,
    		space= search_space,
    		algo=tpe.suggest,
            #algo = rand.suggest,
    		max_evals=args.iterations,
    		trials = trials)


##* outputting trial results

print("Best Parameters: {}".format(best))

with open((model_check_point + '/best_params.json'), 'w', encoding='utf-8') as f:
    json.dump(best, f, default=int)


#plotting.main_plot_history(trials)
#plt.savefig((model_check_point +'/loss_history.png'))

############################################################################
# Evaluating on test data
############################################################################
n_tasks = len(test_dataset.tasks)
model = dc.models.AttentiveFPModel(n_tasks = n_tasks,
                                   mode = 'regression',
                                   model_dir = model_check_point,
                                   batch_size = batch_size,
                                   
                                   ##* AttentiveFP Search Arguments
                                   num_layers = best['num_layers'],
                                   graph_feat_size = best['graph_feat_size'],
                                   dropout = best['dropout'],
                                   learning_rate = best['learning_rate'],
                                   weight_decay = best['weight_decay_penalty'],
)


#print("*** restoring model for evaluation***")
#model.restore(model_dir=model_check_point)

print("*** Training model based on best parameters ***")
model.fit(train_dataset, nb_epoch=epochs)



print('making predictions on test...')
preds = model.predict(test_dataset)
preds = np.squeeze(preds)

df = pd.DataFrame({'actual_RT':test_dataset.y, 'pred_RT':preds})
y_true = df['actual_RT']
y_pred = df['pred_RT']


r2_score = r2_score(y_true, y_pred)
print(f'R2 score: {r2_score}')

mse_score = mean_squared_error(y_true, y_pred, squared=True)
print(f'MSE Score: {mse_score}')

rmse_score = mean_squared_error(y_true, y_pred, squared=False)
print(f'RMSE Score: {rmse_score}')

mae_score = mean_absolute_error(y_true, y_pred)
print(f'MAE Score: {mae_score}')


score_dict = { 
    "R2 Score" : r2_score,
    "MSE Score" : mse_score,
    "RMSE Score" : rmse_score,
    "MAE Score" : mae_score}

with open((model_check_point + '/test_scores.json'), 'w', encoding='utf-8') as f:
    json.dump(score_dict, f, default=int)

print(df.head())
pred_df_path = model_check_point+'/test_preds.csv'
df.to_csv(pred_df_path, index = False)
print(f'saved predictions to {pred_df_path}')

print("\n*** Hyperparameter Optimization is done ***\n")
