# multistep-retrieve-summarize

## Debugging `fairseq-train`
* `fairseq-train` is a `py` file; run `which fairseq-train` to get location.
* Once invoked, calls `def cli_main` in <https://github.com/pytorch/fairseq/blob/master/fairseq_cli/train.py>
* * This gets all the cli args, which are then passed to `distributed_utils.call_main` in <https://github.com/pytorch/fairseq/blob/master/fairseq/distributed_utils.py>
* * This checks for some args, and then calls `distributed_main` which in-turn calls the `def main` function in `fairseq_cli/train.py` -- that is the main training routine.
* In `train.py` the `main` function uses the args to create the `Task` and `Model` which are sent to create a `Trainer` object using <https://github.com/pytorch/fairseq/blob/master/fairseq/trainer.py#L39>
* * `Trainer` sets up the devices, params, etc, required for parallel training and returns a `Trainer` object to the `main` function in `train.py`
* `train.py` will now setup some stuff to start the training -- loading from the last checkpoint, epochs, meters, etc.
* The main training loop will be a `while` routine that checks the `max_epochs` and `lr` learning rate. Default `max_epochs` is infinity ...
* * Inside the loop, each step is 1 epoch. Every epoch (wrapped inside the `epoch_iter`) is sent to `def train` in the same file.
* * * In `def train` you iterate over the samples in the epoch, and call `trainer.train_step(samples)` -- this method runs forward+backward+param_update.
* * * * `trainer.train_step` internally calls `self.task.train_step(sample, model, optimizer, ...)`
* * * * Here, the `task` is a FairseqTask (or can be one of the specific Translation, Classification, LM tasks). The `train_step` function uses native PyTorch to run forward, backward passes.
```python
def train_step(...):
    """Docstring ..."
    model.train()  # This changes the model from `eval` mode to `train` mode!!!
    model.set_num_updates(update_num)
    with torch.autograd.profiler.record_function("forward"):
        loss, sample_size, logging_output = criterion(model, sample)
    if ignore_grad:
        loss *= 0
    with torch.autograd.profiler.record_function("backward"):
        optimizer.backward(loss)
    return loss, sample_size, logging_output
```
* * * * Sample loss is returned at the end, to the `self.task.train_step`
* * * The `train` function in `train.py` finishes all samples in the epoch. After completing the epoch, it logs some stats, resets some meters and returns the epoch losses with `should_stop` to the `while` training loop in `def main` in the same file.
* * `if should_stop` then the `while` loop breaks, else it continues till the training completes.
* `def main` ends with logging a `done training` message.

