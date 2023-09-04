
import pandas as pd 
import numpy as np
import os


##########
## ARGS
#############

path_to_input_data = '../../data/rt_data3_SMILES_filtered.csv'

nr_of_splits = 20

output_dir = '../../data/pub_data/'


#########################################*    

# Reading the data
working_data = pd.read_csv(path_to_input_data)


######* creating timesplits based on 'CREATE_DATE' column

time_sorted_df = working_data.sort_values('CREATE_DATE', ascending=True)

timesplits = np.array_split(time_sorted_df,nr_of_splits)

for i, df in enumerate(timesplits):
    df['timesplit'] = i 

working_data = pd.concat(timesplits)


######* dividing the splits in half; first half is t0 (i.e. traning data); second half will be time-series data

train_selection = list(range(nr_of_splits/2))

train_time = working_data.loc[working_data['timesplit'].isin(train_selection)]

test_time = working_data.loc[~working_data['timesplit'].isin(train_selection)]


#######* saving the splits 


target_dir = os.path.join(output_dir,f'timesplits/timesplit_{nr_of_splits}/splits')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

##* definfing t0 (i.e. training data)
train_time.to_csv(os.path.join(target_dir, 'train_t0.csv'), index=False)


##* defining the 'test' datasets by timesplit
for i,split in enumerate(test_time.timesplit_20.unique()):
    t = f't{(+i+1)}'
    #print(t, split)
    
    subset = test_time[test_time['timesplit'] == split]
    subset.to_csv(os.path.join(target_dir, f'test_{t}.csv'))
    

print('*** Timesplit Done! ***')
