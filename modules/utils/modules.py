
def print_progress_log(epoch: int, logs: dict):
    print(f'\x1b[2K\rEpoch {epoch:3}:', "".join(f" [{key}]{value:5.3f}" for key, value in logs.items()))
