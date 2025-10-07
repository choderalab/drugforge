"""
Class for handling early stopping in training.
"""

from copy import deepcopy

import numpy as np


def _sanitize_loss(loss):
    """
    Helper function for the ES classes to make sure that they receive a single float as
    their loss value. If an iterable of floats is passed, the mean loss will be returned

    Parameters
    ----------
    loss : Union[float, List[float], np.ndarray, torch.Tensor]
        Loss value(s)

    Returns
    -------
    float
        Sanitized loss value
    """
    try:
        # This should work for common types of numeric values (single float, list,
        #  tensor, etc of floats)
        return np.asarray(loss).mean()
    except Exception:
        raise ValueError(f"Bad value passed for loss: {loss}")


class BestEarlyStopping:
    """
    Class for handling early stopping in training based on improvement over best loss.
    """

    def __init__(self, patience, burnin=0):
        """
        Parameters
        ----------
        patience : int
            The maximum number of epochs to continue training with no improvement in the
            val loss. If not given, no early stopping will be performed
        burnin : int, optional
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.patience = patience
        self.burnin = burnin

        # Variables to track early stopping
        self.counter = 0
        self.best_loss = None
        self.best_wts = None
        self.best_epoch = 0

    def check(self, epoch, loss, wts_dict):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Model loss from the current epoch of training
        wts_dict : dict
            Weights dict from Pytorch for keeping track of the best model

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)

        # If this is the first epoch, just set internal variables and return
        if self.best_loss is None:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            return False

        # Update best loss and best weights
        if loss < self.best_loss:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            self.best_epoch = epoch

            # Reset counter
            self.counter = 0

            # Keep training
            return False

        # Increment counter and check for stopping
        self.counter += 1
        if (self.counter >= self.patience) and (epoch >= self.burnin):
            return True

        return False


class ConvergedEarlyStopping:
    """
    Class for handling early stopping in training based on whether loss is still
    changing. Check that the mean difference of the past n losses from the average of
    those losses is within tolerance.
    """

    def __init__(self, n_check, divergence, burnin=0):
        """
        Parameters
        ----------
        n_check : int
            Number of past epochs to keep track of when calculating divergence
        divergence : float
            Max allowable difference from the mean of the losses
        burnin : int, optional
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.n_check = n_check
        self.divergence = divergence
        self.burnin = burnin

        # Variables to track early stopping
        self.losses = []

    def check(self, epoch, loss):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Loss from the previous training epoch

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)

        # Add most recent loss
        self.losses += [loss]

        # Don't have enough samples yet, so keep training
        if len(self.losses) < self.n_check:
            return False

        # Full loss buffer, so get rid of earliest loss
        if len(self.losses) > self.n_check:
            self.losses = self.losses[1:]

        # Check for early stopping
        mean_loss = np.mean(self.losses)
        all_abs_diff = np.abs(np.asarray(self.losses) - mean_loss)

        return (np.mean(all_abs_diff) < self.divergence) and (epoch >= self.burnin)


class PatientConvergedEarlyStopping:
    """
    Class for handling early stopping in training based on whether loss is still
    changing, with patience. Check that the mean difference of the past n losses from
    the average of those losses is within tolerance, then wait to make sure it's not a
    temporary plateau.
    """

    def __init__(self, n_check, divergence, patience, burnin=0):
        """
        Parameters
        ----------
        n_check : int
            Number of past epochs to keep track of when calculating divergence
        divergence : float
            Max allowable difference from the mean of the losses
        patience : int
            The maximum number of epochs to wait after convergence
        burnin : int, optional
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.n_check = n_check
        self.divergence = divergence
        self.patience = patience
        self.burnin = burnin

        # Variables to track early stopping
        # Window of losses to check for convergence
        self.losses = []
        # Tracker for if we've reached convergence
        self.converged = False
        # Tracker for how many epochs it's been since we've converged
        self.counter = 0
        # Loss val at convergence
        self.converged_loss = None
        # Model weights at convergence
        self.converged_wts = None
        # Epoch we reached convergence
        self.converged_epoch = 0

    def check(self, epoch, loss, wts_dict):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Model loss from the current epoch of training
        wts_dict : dict
            Weights dict from Pytorch for keeping track of the best model

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)

        # Add most recent loss
        self.losses += [loss]

        # Don't have enough samples yet, so keep training
        if len(self.losses) < self.n_check:
            return False

        # Full loss buffer, so get rid of earliest loss
        if len(self.losses) > self.n_check:
            self.losses = self.losses[1:]

        # Check for early stopping
        mean_loss = np.mean(self.losses)
        all_abs_diff = np.abs(np.asarray(self.losses) - mean_loss)

        converged = np.mean(all_abs_diff) < self.divergence

        if converged:
            if self.converged:
                self.counter += 1
                print("converged patience counter", self.counter, flush=True)
                if (self.counter >= self.patience) and (epoch >= self.burnin):
                    return True
            else:
                self.converged = True

                self.converged_loss = loss
                # Need to deepcopy so it doesn't update with the model weights
                self.converged_wts = deepcopy(wts_dict)
                self.converged_epoch = epoch
        elif self.converged:
            # Reset everything
            self.converged = False
            self.counter = 0
            self.converged_loss = None
            self.converged_wts = None
            self.converged_epoch = 0

        return False


class ThresholdEarlyStopping:
    """
    Class for handling early stopping in training based on whether loss has been below
    a certain threshold for some number of epochs.
    """

    def __init__(self, threshold, patience, burnin=0):
        """
        Parameters
        ----------
        threshold : float
            Loss below which to stop model training
        patience : ing
            Number of epochs to wait once loss has dipped below threshold to make sure
            it stays there
        burnin : int, optional
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.burnin = burnin

        # Variables to track early stopping
        self.converged_epochs = 0

    def check(self, epoch, loss):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        epoch : int
            Current training epoch
        loss : float
            Model loss from the current epoch of training

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)

        if loss > self.threshold:
            self.converged_epochs = 0
            return False

        self.converged_epochs += 1
        if (self.converged_epochs == self.patience) and (epoch >= self.burnin):
            return True
        return False


# Losses from the Tricks of the Trade book
class GeneralizationLossEarlyStopping:
    """
    Class for stopping based on the relative increase of val loss at epoch t from
    lowest val loss up to epoch t. GL_alpha loss in the book.
    GL is defined as 100 * (Err_min / Err(t) - 1)
    """

    def __init__(self, alpha, burnin=0):
        """
        Parameters
        ----------
        alpha : float
            Relative increase threshold
        burnin : int, default = 0
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.alpha = alpha
        self.burnin = burnin

        # Variables to track early stopping
        self.best_loss = None
        self.best_wts = None
        self.best_epoch = 0

    def check(self, epoch, loss, wts_dict):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Model loss from the current epoch of training
        wts_dict : dict
            Weights dict from Pytorch for keeping track of the best model

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)

        # If this is the first epoch, just set internal variables and return
        if self.best_loss is None:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            return False

        # Update best loss and best weights
        if loss < self.best_loss:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            self.best_epoch = epoch

        if epoch < self.burnin:
            return False

        # Calculate generalization loss and check stopping criteria
        generalization_loss = 100 * (loss / self.best_loss - 1)
        return generalization_loss > self.alpha


class ProgressQuotientEarlyStopping:
    """
    Class for stopping based on quotient of generalization loss and training progress.
    PQ_alpha in the book. This criterion is calculated after every k epochs, and
    interstitial epochs will automatically not be stopped at.
    """

    def __init__(self, alpha, k, burnin=0):
        """
        Parameters
        ----------
        alpha : float
            Quotient threshold
        k : int
            Length of training strip to evaluate at the end of
        burnin : int, default = 0
            If given, ensure that at least this many epochs of training have been done
            before we stop
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.burnin = burnin

        # Variables to track early stopping
        self.best_loss = None
        self.best_wts = None
        self.best_epoch = 0
        self.strip_train_losses = []

    def check(self, epoch, loss, wts_dict, train_loss):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Model loss from the current epoch of training
        wts_dict : dict
            Weights dict from Pytorch for keeping track of the best model

        Returns
        -------
        bool
            Whether to stop training
        """
        # Make sure we've got a reasonable value for loss
        loss = _sanitize_loss(loss)
        train_loss = _sanitize_loss(train_loss)

        # If this is the first epoch, just set internal variables and return
        if self.best_loss is None:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)

            self.strip_train_losses += [train_loss]
            return False

        # Update best loss and best weights
        if loss < self.best_loss:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            self.best_epoch = epoch

            self.strip_train_losses += [train_loss]
            return False

        # Make sure we're at the end of a training strip
        if (epoch < self.burnin) or (epoch < self.k) or (epoch % self.k != 0):
            self.strip_train_losses += [train_loss]
            return False

        # Calculate generalization loss and progress
        generalization_loss = 100 * (loss / self.best_loss - 1)
        progress = 1000 * (
            np.mean(self.strip_train_losses) / np.min(self.strip_train_losses) - 1
        )
        self.strip_train_losses = []

        return generalization_loss / progress > self.alpha
