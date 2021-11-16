from torch import nn
from torch import optim
from specs import *
import dlc_practical_prologue as prologue
plot = True
try:
    import matplotlib.pyplot as plt
except:
    print('\nRunning without matplotlib.')
    plot = False

# Training parameters
n_samples   =  1000
n_epochs    =   100
n_runs      =    10
btch_size   =    50
optimizers  = [ optim.SGD, optim.Adam]
optim_names = ['SGD',     'Adam'     ]
learn_rates = [ 1e-1,      5e-3      ]

# General parameters (always have Net_0() as a first element for the dashed line on the graphs)
models  = [Net_0(), Net_1(),
           Net_2(), Net_2(mode='never'), Net_2(mode='once'), Net_2(mode='always'),
           Net_3(), Net_3(mode='never'), Net_3(mode='once'), Net_3(mode='always')]
use_gpu = True

# Load and pre-process the data set
train_i, train_t, train_c, test_i, test_t, test_c = prologue.generate_pair_sets(n_samples)
mean, std = train_i.mean(), train_i.std()
train_i.sub_(mean).div_(std)
test_i. sub_(mean).div_(std)

# Re-arrange the input samples for simpler code
train_i = (train_i[:,0,:,:].unsqueeze(1), train_i[:,1,:,:].unsqueeze(1))
test_i  = (test_i[ :,0,:,:].unsqueeze(1), test_i[ :,1,:,:].unsqueeze(1))

# Map variables to GPU if detected
if use_gpu and torch.cuda.is_available():
    print('Using GPU!! GPU: ' + torch.cuda.get_device_name(0))
    models  = [model. to('cuda') for model in models ]
    train_i = [i.     to('cuda') for i     in train_i]
    test_i  = [i.     to('cuda') for i     in test_i ]
    train_t = train_t.to('cuda')
    test_t  = test_t. to('cuda')
    train_c = train_c.to('cuda')
    test_c  = test_c. to('cuda')

# Try different optimizers
for index, optimizer in enumerate(optimizers):

    # Train the models over several runs
    print('\nUsing '+optim_names[index]+' optimizer:')
    model_perfs = torch.zeros((len(models), 2, n_runs, n_epochs))
    for i, model in enumerate(models):

        # Initialize criterion and optimizer
        print('\n    Training the model ' + model.description + '...')
        crit = nn.CrossEntropyLoss()
        opt  = optimizer(model.parameters(), lr = learn_rates[index])
        for run in range(n_runs):

            # Actual training part
            train_errors, test_errors = train_model(model, crit, opt, n_epochs,
                btch_size, train_i, train_t, train_c, test_i, test_t, test_c)

            # Store the data and print the error after each run
            model_perfs[i, 0, run, :] = torch.FloatTensor(train_errors)
            model_perfs[i, 1, run, :] = torch.FloatTensor(test_errors )
            print('        Run: %1i - Final train error: %5.2f %% - Final test error: %5.2f %%' % \
                (run, train_errors[-1], test_errors[-1]))

            # Reset the model parameters for the next run
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.reset_parameters()

            # Re-shuffle the training set after each run
            shuffle =  torch.randperm(train_i[0].size(0))
            train_i = [train[shuffle] for train in train_i]
            train_t =  train_t[shuffle]
            train_c =  train_c[shuffle]

    # Plot model performance, with standard deviation
    if plot:
        means = model_perfs.mean(dim=2).numpy()
        stds  = model_perfs.std( dim=2).numpy()
        plt.figure('Results using '+optim_names[index]+' optimizer')
        for i, model in enumerate(models):
            plt.subplot(int(len(models)/4.0-0.1)+1, min(len(models), 4), i+1+2*int(i>=2))
            plt.plot((0, n_epochs), (means[0,1,-1], means[0,1,-1]), 'k--')
            plt.plot(means[i,0], color='m', label='training')
            plt.plot(means[i,1], color='r', label='testing' )
            plt.fill_between(range(n_epochs), means[i,0]-stds[i,0]/2.0, means[i,0]+stds[i,0]/2.0, facecolor='m', alpha=0.5)
            plt.fill_between(range(n_epochs), means[i,1]-stds[i,1]/2.0, means[i,1]+stds[i,1]/2.0, facecolor='r', alpha=0.5)
            plt.ylim((0, 60))
            plt.title(model.description)
            plt.legend(loc='upper right')

# Show figures and goodbye message
if plot:
    plt.show()
print('\nSimulation finished!\n')