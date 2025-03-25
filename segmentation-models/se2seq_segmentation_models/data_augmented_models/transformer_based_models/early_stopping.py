class EarlyStopping:
   """Early stops the training if validation loss doesn't improve after a given patience."""
   def __init__(self, plateau_patience=5, worsen_patience=3, min_delta=0, verbose=True):
       """
       Args:
           plateau_patience (int): How long to wait when loss plateaus before stopping.
           worsen_patience (int): How long to wait when loss worsens before stopping. 
           min_delta (float): Minimum change in monitored value to qualify as an improvement.
           verbose (bool): If True, prints a message for each validation loss change.
       """
       self.plateau_patience = plateau_patience
       self.worsen_patience = worsen_patience
       self.min_delta = min_delta
       self.verbose = verbose
       self.plateau_counter = 0
       self.worsen_counter = 0
       self.best_loss = None
       self.early_stop = False

   def __call__(self, val_loss, checkpoint_data, save_checkpoint_fn):
       """
       Check for stopping criteria based on validation loss.
       Args:
           val_loss (float): Validation loss for current epoch.
           checkpoint_data (dict): Data to save in checkpoint.
           save_checkpoint_fn (callable): Function to save checkpoint.
       """
       if self.best_loss is None:
           # First epoch
           self.best_loss = val_loss
           save_checkpoint_fn(checkpoint_data, is_best=True)
           if self.verbose:
               print(f"Initial best loss set to {val_loss:.6f}.")
           return

       if val_loss < self.best_loss:  # Loss has improved
           if self.verbose:
               print(f"Improvement: {self.best_loss:.6f} -> {val_loss:.6f}")
           self.best_loss = val_loss  # Update best loss
           save_checkpoint_fn(checkpoint_data, is_best=True)
           self.plateau_counter = 0
           self.worsen_counter = 0

       elif val_loss > self.best_loss:  # Loss has worsened
           self.worsen_counter += 1
           self.plateau_counter += 1
           if self.verbose:
               print(f"Loss increased: {self.best_loss:.6f} -> {val_loss:.6f}. Counter: {self.worsen_counter}/{self.worsen_patience}")
           if self.worsen_counter >= self.worsen_patience:
               self.early_stop = True
               if self.verbose:
                   print("Early stopping triggered due to loss increasing.")

       else:  # val_loss == self.best_loss (exact plateau)
           self.plateau_counter += 1
           if self.verbose:
               print(f"Loss plateaued at {val_loss:.6f}. Counter: {self.plateau_counter}/{self.plateau_patience}")

       # Check plateau counter separately as it applies to both worsening and plateaus
       if self.plateau_counter >= self.plateau_patience:
           self.early_stop = True
           if self.verbose:
               print("Early stopping triggered due to lack of improvement.")

   def reset(self):
       """Resets all counters and flags."""
       self.plateau_counter = 0
       self.worsen_counter = 0
       self.best_loss = None
       self.early_stop = False