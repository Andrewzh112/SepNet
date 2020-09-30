from sepnet.config import Config
from sepnet.train import Trainer
from sepnet.model import MobileUnet
from sepnet.data import SepDataset


def main():
    config = Config()
    trn_dataset = SepDataset(filter_imgs=True)
    tst_dataset = SepDataset(filter_imgs=True,
                             statistics=trn_dataset.statistics,
                             img_path='test_images')
    model = MobileUnet(3)
    trainer = Trainer(model, config, trn_dataset, tst_dataset)
    trainer.train(1)


if __name__ == '__main__':
    main()
