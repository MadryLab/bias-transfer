import torchvision
import src.loaders as loaders_util

topil = torchvision.transforms.ToPILImage()

def save_example_images(loader, dataset, filename):
    if len(loader) > 0:
        invTrans = loaders_util.inv_norm(dataset)
        for X,_ in loader:
            break
        grid = torchvision.utils.make_grid(invTrans(X), nrow=50)
        topil(grid).save(filename)


class LambdaLoader:
    '''
    This is a class that allows one to apply any given (fixed) 
    transformation to the output from the loader in *real-time*.
    For instance, you could use for applications such as custom 
    data augmentation and adding image/label noise.
    Note that the LambdaLoader is the final transformation that
    is applied to image-label pairs from the dataset as part of the
    loading process---i.e., other (standard) transformations such
    as data augmentation can only be applied *before* passing the
    data through the LambdaLoader.
    For more information see :ref:`our detailed walkthrough <using-custom-loaders>`
    '''

    def __init__(self, loader, func):
        '''
        Args:
            loader (PyTorch dataloader) : loader for dataset (*required*).
            func (function) : fixed transformation to be applied to 
                every batch in real-time (*required*). It takes in 
                (images, labels) and returns (images, labels) of the 
                same shape.
        '''
        self.data_loader = loader
        self.loader = iter(self.data_loader)
        self.func = func

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return self

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)

    def __next__(self):
        try:
            im, targ = next(self.loader)
        except StopIteration as e:
            self.loader = iter(self.data_loader)
            raise StopIteration

        return self.func(im, targ)

    def __getattr__(self, attr):
        return getattr(self.data_loader, attr)