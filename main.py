from sepnet.config import Config
from sepnet.train import Trainer
from sepnet.model import MobileUnet
from sepnet.data import SepDataset


def main():
    config = Config()
    dataset = SepDataset(filter_imgs=True)
    model = MobileUnet(3)
    trainer = Trainer(model, config, dataset, dataset)
    trainer.train()


if __name__ == '__main__':
    main()
