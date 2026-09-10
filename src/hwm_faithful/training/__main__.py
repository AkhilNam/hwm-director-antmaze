from hwm_faithful.training.cli import main
import torch

if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
