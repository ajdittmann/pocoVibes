from typing import Union, Optional, Tuple, Dict, List

import numpy as np
import copy
import time 
from collections import OrderedDict
import zuko
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .tools import torch_double_to_float


def _batch_indices(n, batch_size, shuffle):
    """
    Yield index objects that split ``n`` samples into mini-batches of size
    ``batch_size`` (the last batch may be smaller), in random order if ``shuffle``.
    """
    if shuffle:
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            yield perm[start:start + batch_size]
    else:
        for start in range(0, n, batch_size):
            yield slice(start, start + batch_size)


def _clone_state_dict(state_dict):
    """
    Copy a module ``state_dict`` (tensors are cloned, other entries deep-copied).
    """
    cloned = OrderedDict()
    for k, v in state_dict.items():
        cloned[k] = v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        cloned._metadata = copy.deepcopy(metadata)
    return cloned

class Flow:
    """
    Normalizing flow model.

    Parameters
    ----------
    n_dim : ``int``
        Number of dimensions of the distribution to be modeled.
    flow : ``zuko.flows.Flow`` or str, optional
        Normalizing flow model. Default: ``nsf3``.

    Attributes
    ----------
    n_dim : ``int``
        Number of dimensions of the distribution to be modeled.
    flow : ``zuko.flows.Flow``
        Normalizing flow model.
    transform : ``zuko.transforms.Transform``
        Transformation object.
    
    Examples
    --------
    >>> import torch
    >>> import pocomc
    >>> flow = pocomc.Flow(2)
    >>> x = torch.randn(100, 2)
    >>> u, logdetj = flow(x)
    >>> x_, logdetj_ = flow.inverse(u)
    >>> log_prob = flow.log_prob(x)
    >>> x_, log_prob_ = flow.sample(100)
    >>> history = flow.fit(x)
    """

    def __init__(self, n_dim, flow='nsf3'):
        self.n_dim = n_dim

        def next_power_of_2(n):
            return 1 if n == 0 else 2**(n - 1).bit_length()
        
        n_hidden = np.maximum(next_power_of_2(3*n_dim), 32)

        if flow == 'maf3':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=3, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'maf6':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=6, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'maf12':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=12, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'nsf3':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=3, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif flow == 'nsf6':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=6, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif flow == 'nsf12':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=12, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif isinstance(flow, zuko.flows.Flow):
            self.flow = flow
        else:
            raise ValueError('Invalid flow type. Choose from: maf3, maf6, maf12, nsf3, nsf6, nsf12, or provide a zuko.flows.Flow object.')

    @property
    def transform(self):
        """
        Transformation object.
        """
        return self.flow().transform

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.
        Inputs are transformed from the original (relating to the distribution to be modeled) to the latent space.

        Parameters
        ----------
        x : ``torch.Tensor``
            Samples to transform.
        Returns
        -------
        u : ``torch.Tensor``
            Transformed samples in latent space with the same shape as the original space inputs.
        """
        x = torch_double_to_float(x)
        return self.transform.call_and_ladj(x)

    def inverse(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation.
        Inputs are transformed from the latent to the original space (relating to the distribution to be modeled).
        
        Parameters
        ----------
        u : ``torch.Tensor``
            Samples to transform.
        Returns
        -------
        x : ``torch.Tensor``
            Transformed samples in the original space with the same shape as the latent space inputs.
        """
        u = torch_double_to_float(u)
        x, logdetj = self.transform.inv.call_and_ladj(u)
        return x, logdetj

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples.
        
        Parameters
        ----------
        x : ``torch.Tensor``
            Input samples
        Returns
        -------
        Log-probability of samples.
        """
        x = torch_double_to_float(x)
        return self.flow().log_prob(x)

    def sample(self, size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw random samples from the normalizing flow.

        Parameters
        ----------
        size : ``int``
            Number of samples to generate. Default: 1.
        Returns
        -------
        samples, log_prob : ``tuple``
            Samples as a ``torch.Tensor`` with shape ``(size, n_dimensions)`` and log probability values with shape ``(size, )``.
        """
        x, log_p = self.flow().rsample_and_log_prob((size,))
        return x, log_p

    def fit(self,
            x,
            weights=None,
            validation_split=0.0,
            epochs=1000,
            batch_size=1000,
            patience=20,
            learning_rate=1e-3,
            weight_decay=0,
            laplace_scale=None,
            gaussian_scale=None,
            annealing=True,
            noise=None,
            shuffle=True,
            clip_grad_norm=1.0,
            verbose=0,
            ):
        """

        Parameters
        ----------
        x : ``torch.Tensor``
            Input samples.
        weights : ``torch.Tensor``, optional
            Weights for each sample. Default: ``None``.
        validation_split : ``float``, optional
            Fraction of samples to use for validation. Default: 0.0.
        epochs : ``int``, optional
            Number of epochs. Default: 1000.
        batch_size : ``int``, optional
            Batch size. Default: 1000.
        patience : ``int``, optional
            Number of epochs without improvement before early stopping. Default: 20.
        learning_rate : ``float``, optional
            Learning rate. Default: 1e-3.
        weight_decay : ``float``, optional
            Weight decay. Default: 0.
        laplace_scale : ``float``, optional
            Laplace regularization scale. Default: ``None``.
        gaussian_scale : ``float``, optional
            Gaussian regularization scale. Default: ``None``.
        annealing : ``bool``, optional
            Whether to use learning rate annealing. Default: ``True``.
        noise : ``float``, optional
            Noise scale. Default: ``None``.
        shuffle : ``bool``, optional
            Whether to shuffle samples. Default: ``True``.
        clip_grad_norm : ``float``, optional
            Maximum gradient norm. Default: 1.0.
        verbose : ``int``, optional
            Verbosity level. Default: 0.

        Returns
        -------
        history : ``dict``
            Dictionary with loss history.

        Examples
        --------
        >>> import torch
        >>> import pocomc
        >>> flow = pocomc.Flow(2)
        >>> x = torch.randn(100, 2)
        >>> history = flow.fit(x)    
        """
        x = torch_double_to_float(x)

        n_samples, n_dim = x.shape

        if shuffle:
            rand_indx = torch.randperm(n_samples)
            x = x[rand_indx]
            if weights is not None:
                weights = weights[rand_indx]

        if noise is not None:
            dists = torch.cdist(x, x)
            # Mask out the diagonal
            dists.fill_diagonal_(float('inf'))
            min_dists = torch.min(dists, dim=1).values
            mean_min_dist = torch.mean(min_dists)

        # Mini-batches are drawn by direct tensor indexing (equivalent to a
        # DataLoader over a TensorDataset, without per-sample collation overhead)
        weights_train = None
        weights_valid = None
        if validation_split > 0.0:
            # The first (1 - validation_split) of the samples are used for training and the
            # remaining validation_split for validation
            n_split = int((1.0 - validation_split) * n_samples)
            x_train = x[:n_split]
            x_valid = x[n_split:]
            if weights is not None:
                weights_train = weights[:n_split]
                weights_valid = weights[n_split:]
            validation = True
        else:
            x_train = x
            if weights is not None:
                weights_train = weights
            validation = False
        n_train = len(x_train)
        n_valid = len(x_valid) if validation else 0

        optimizer = torch.optim.AdamW(self.flow.parameters(), 
                                      learning_rate,
                                      weight_decay=weight_decay,
                                      )

        if annealing:
            scheduler = ReduceLROnPlateau(optimizer, 
                                          mode='min',
                                          factor=0.2,
                                          patience=patience, 
                                          threshold=0.0001, 
                                          threshold_mode='abs', 
                                          min_lr=1e-6,
                                         )

        history = dict()  # Collects per-epoch loss
        history['loss'] = []
        history['val_loss'] = []
        if validation:
            monitor = 'val_loss'
        else:
            monitor = 'loss'

        best_epoch = 0
        best_loss = np.inf
        best_model = _clone_state_dict(self.flow.state_dict())

        start_time_sec = time.time()

        for epoch in range(epochs):
            self.flow.train()
            train_loss = 0.0

            for idx in _batch_indices(n_train, batch_size, shuffle):

                optimizer.zero_grad()
                x_batch = x_train[idx]
                if noise is None:
                    x_ = x_batch
                else:
                    x_ = x_batch + noise * mean_min_dist * torch.randn_like(x_batch)
                if weights is None:
                    loss = -self.flow().log_prob(x_).sum()
                else:
                    w_batch = weights_train[idx]
                    loss = -self.flow().log_prob(x_) * w_batch * 1000.0
                    loss = loss.sum() / w_batch.sum()

                if laplace_scale is not None or gaussian_scale is not None:
                    loss -= regularization_loss(self.flow, laplace_scale, gaussian_scale)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), clip_grad_norm)
                optimizer.step()

                train_loss += loss.data.item()

            train_loss = train_loss / n_train

            history['loss'].append(train_loss)
            
            if validation:
                self.flow.eval()
                val_loss = 0.0

                # No gradients are needed for the validation loss
                with torch.no_grad():
                    for idx in _batch_indices(n_valid, batch_size, shuffle):

                        x_batch = x_valid[idx]
                        if noise is None:
                            x_ = x_batch
                        else:
                            x_ = x_batch + noise * mean_min_dist * torch.randn_like(x_batch)
                        if weights is None:
                            loss = -self.flow().log_prob(x_).sum()
                        else:
                            w_batch = weights_valid[idx]
                            loss = -self.flow().log_prob(x_) * w_batch * 1000.0
                            loss = loss.sum() / w_batch.sum()
                        
                        if laplace_scale is not None or gaussian_scale is not None:
                            loss -= regularization_loss(self.flow, laplace_scale, gaussian_scale)

                        val_loss += loss.data.item()

                val_loss = val_loss / n_valid

                history['val_loss'].append(val_loss)
        
            if annealing and validation:
                scheduler.step(val_loss)
            elif annealing:
                scheduler.step(train_loss)

            if verbose > 1:
                try:
                    print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))
                except:  # TODO specify type of exception
                    print('Epoch %3d/%3d, train loss: %5.2f' % (epoch + 1, epochs, train_loss))

            # Monitor loss
            if history[monitor][-1] < best_loss:
                best_loss = history[monitor][-1]
                best_epoch = epoch
                best_model = _clone_state_dict(self.flow.state_dict())

            if epoch - best_epoch >= int(1.5 * patience):
                self.flow.load_state_dict(best_model)
                if verbose > 0:
                    print('Finished early after %3d epochs' % best_epoch)
                    print('Best loss achieved %5.2f' % best_loss)
                break
        else:
            # All epochs used without early stopping: keep the best model, not the last one
            if best_epoch != epochs - 1:
                self.flow.load_state_dict(best_model)
        
        if verbose > 0:
            end_time_sec = time.time()
            total_time_sec = end_time_sec - start_time_sec
            time_per_epoch_sec = total_time_sec / epochs
            print()
            print('Time total:     %5.2f sec' % total_time_sec)
            print('Time per epoch: %5.2f sec' % time_per_epoch_sec)

        return history


def regularization_loss(model, laplace_scale=None, gaussian_scale=None):
    """
    Compute regularization loss.

    Parameters
    ----------
    model : ``zuko.flows.Flow``
        Normalizing flow model.
    laplace_scale : ``float``, optional
        Laplace regularization scale. Default: ``None``.
    gaussian_scale : ``float``, optional
        Gaussian regularization scale. Default: ``None``.
    
    Returns
    -------
    Regularization loss.
    """
    total_laplace = 0.0
    total_gaussian = 0.0

    for i, transform in enumerate(model.transforms):
        if hasattr(transform, "hyper"):
            for parameter_name, parameter in transform.hyper.named_parameters():
                if parameter_name.endswith('weight'):
                    if laplace_scale is not None:
                        total_laplace += parameter.abs().sum()
                    if gaussian_scale is not None:
                        total_gaussian += parameter.square().sum()
    
    total = 0.0
    if laplace_scale is not None:
        total += - total_laplace / laplace_scale
    if gaussian_scale is not None:
        total += - total_gaussian / (2.0 * gaussian_scale**2.0)

    return total
