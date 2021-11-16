import torch
from torch import nn
from torch import optim


# Take any network, train it on the training data, test it on the testing data, return the final errors 
def train_model(model, crit, optimizer, n_epochs, btch_size, train_i, train_t, train_c, test_i, test_t, test_c):

    # Go through the training set for several epochs
    train_errors, test_errors = ([], [])
    for epoch in range(n_epochs):

        # Update the learning schedule
        model.learning_scheduler(epoch, n_epochs)

        # Run one batch and apply gradient descent
        n_train_errors = 0
        for b in range(0, train_i[0].size(0) - (train_i[0].size(0)%btch_size), btch_size):

            # Take a batch in the training set and compute the output
            trains = (train_i[0].narrow(0, b, btch_size), train_i[1].narrow(0, b, btch_size))
            output = model(trains)

            # Compute the loss associated to the model's output
            loss, n_errors  = run_train_batch(model, output, crit, train_t, train_c, b, btch_size)
            n_train_errors += n_errors

            # Update the weights
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Print stuff to monitor the progress on training and testing performance
        train_error = 100*float(n_train_errors)/train_i[0].shape[0]
        test_error  = compute_error_percent(model, test_i, test_t, test_c)
        print('        Train error: %5.2f %% - Test error: %5.2f %% \033[1A' % (train_error, test_error))
        train_errors.append(train_error)
        test_errors .append(test_error )

    # Return the last testing and training errors
    return train_errors, test_errors


# Run one batch and return the loss and number of error made in the batch
def run_train_batch(model, output, crit, train_t, train_c, b, btch_size, loss=0):

    # All network containing two number classifiers and a comparator classifier
    if hasattr(model, 'comp_classifier'):

        target = train_t.narrow(0, b, btch_size)
        # if any(p.requires_grad for p in model.comp_classifier.parameters()):
        if model.loss_basis in ['comp', 'both']:
            loss += crit(output[2], target)
        if model.loss_basis in ['num',  'both']:
        # if any(p.requires_grad for p in model.num_classifier.parameters()):
            target_c = train_c.narrow(0, b, btch_size)
            loss += crit(output[0], target_c[:,0]) + crit(output[1], target_c[:,1])
        n_errors = (output[2].argmax(1) != target).sum()

    # All simpler networks (one classifier), for Net_0() target is train_c
    else:
        if model._get_name() in ['Net_0']:
            target    = train_c.narrow(0, b, btch_size)
            loss     += crit(output[0], target[:,0]) + crit(output[1], target[:,1])
            n_errors  = (output[0].argmax(1) != target[:,0]).sum()/2.0
            n_errors += (output[1].argmax(1) != target[:,1]).sum()/2.0
        else:
            target   = train_t.narrow(0, b, btch_size)
            loss    += crit(output, target)
            n_errors = (output.argmax(1) != target).sum()

    # Return the loss and the number of errors
    return loss, n_errors


# Compute the percentage of error for any model, without gradient
def compute_error_percent(model, i, t, c):

    # Run the model on the whole data without computing gradient
    with torch.no_grad():
        output = model(i)

    # Compute the percentage of errors done by the model
    if hasattr(model, 'comp_classifier'):
        error_percent = 100*float((output[2].argmax(1) != t).sum())/t.shape[0]
    else:
        if model._get_name() in ['Net_0']:
            error_percent  = 100*float((output[0].argmax(1) != c[:,0]).sum())/(2*c.shape[0])
            error_percent += 100*float((output[1].argmax(1) != c[:,1]).sum())/(2*c.shape[0])
        else:
            error_percent = 100*float((output.argmax(1) != t).sum())/t.shape[0]
    
    # Return the percentage of errors
    return error_percent


# Basic network for number classification
class Net_0(nn.Module):

    def __init__(self, num_classes=10):
        super(Net_0, self).__init__()    

        self.description = self._get_name()
        self.n_feat_1 = 16
        self.n_feat_2 = 64
        self.n_feat_3 = 64
        self.n_feat_4 = 64
        self.features = nn.Sequential(
            nn.Conv2d( 1, self.n_feat_1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.n_feat_1, self.n_feat_2, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))      
        self.num_classifier = nn.Sequential(
            nn.Linear(2*2*self.n_feat_2, self.n_feat_3),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feat_3, self.n_feat_4),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feat_4, num_classes))

    def learning_scheduler(self, epoch, n_epoch):
        pass

    def num_classification(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2*2*self.n_feat_2)
        x = self.num_classifier(x)
        return x

    def forward(self, xs):
        x0 = self.num_classification(xs[0])
        x1 = self.num_classification(xs[1])
        return (x0, x1)


# Net_0 takes both number images as two features and classify straightforwardly 
class Net_1(Net_0):

    def __init__(self, num_classes=10):
        super(Net_1, self).__init__(num_classes)
        
        self.features[0]        = nn.Conv2d(2, self.n_feat_1, kernel_size=3)
        self.num_classifier[-1] = nn.Linear(self.n_feat_4, 2)

    def forward(self, xs):
        return self.num_classification(torch.cat(xs, dim=1))


# Two Net_0, sharing weights, whose outputs go to a final classifier
class Net_2(Net_0):

    def __init__(self, num_classes=10, mode=None):
        super(Net_2, self).__init__(num_classes)

        self.mode = mode
        if mode in ['never']:
            self.loss_basis  = 'both'
            self.description = self._get_name() + ' using both losses'
        elif mode in ['once', 'always']:
            self.loss_basis  = 'num'
            self.description = self._get_name() + ' alternating loss ' + mode
        else:
            self.loss_basis  = 'comp'
            self.description = self._get_name() + ' using only the final loss'
        self.n_feat_5        = 64
        self.comp_classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(2*num_classes, self.n_feat_5),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feat_5, 2))

    def learning_scheduler(self, epoch, n_epochs):

        # After each run, swapping networks need this reset
        if epoch == 0 and self.mode in ['once', 'always']:
            self.loss_basis = 'num'

        else:
            if (self.mode == 'once' and epoch == n_epochs/2) or self.mode == 'always':
                self.loss_basis = 'num' if self.loss_basis == 'comp' else 'comp'

    def forward(self, xs):
        x0 = self.num_classification(xs[0])
        x1 = self.num_classification(xs[1])
        x  = self.comp_classifier(torch.cat((x0, x1), dim=1))
        return (x0, x1, x)


# Two Net_0, not sharing weights, whose outputs go to a final classifier
class Net_3(Net_2):

    def __init__(self, num_classes=10, mode=None):
        super(Net_3, self).__init__(num_classes, mode)

        # Overwrite the weight-sharing classifier
        self.num_classifier = nn.Sequential(Net_0(), Net_0())

    def forward(self, xs):

        x0 = self.num_classifier[0].num_classification(xs[0])
        x1 = self.num_classifier[1].num_classification(xs[1])
        x  = self.comp_classifier(torch.cat((x0, x1), dim=1))
        return (x0, x1, x)
