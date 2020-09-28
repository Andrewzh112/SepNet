import torch
from torch.utils.data import DataLoader
from sepnet.utils import mix_images
# from sepnet.loss import SeperationLoss
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, trn_dataset, tst_dataset):
        self.config = config
        self.model = model
        self.trn_dataset = trn_dataset
        self.tst_dataset = tst_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def train(self):
        model, config, device = self.model, self.config, self.device
        criterion = None  # TODO SeperationLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), **config.opimizer_params
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )
        trn_loader = DataLoader(self.trn_dataset, shuffle=True,
                                pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
        tst_loader = DataLoader(self.tst_dataset, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

        pbar = tqdm(range(config.epochs), total=config.epochs)

        for epoch in pbar:

            def run_epoch(is_train, loader):
                losses = []
                for images in loader:
                    images = images.to(device)
                    image_pairs, mixed_images, ratios = mix_images(images)
                    constructed_images = model(mixed_images)
                    loss = criterion(constructed_images, image_pairs, ratios)
                    losses.append(loss.item())
                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm_clip
                        )
                        optimizer.step()
                        scheduler.step()
                return losses

            model.train()
            train_epoch_loss = run_epoch(is_train=True, loader=trn_loader)
            mean_train_epoch_loss = sum(
                train_epoch_loss) / len(train_epoch_loss)
            logger.info(
                f'Epoch {epoch + 1}/{config.epochs}, \
                    The training loss is {mean_train_epoch_loss:.3f}')

            model.eval()
            test_epoch_loss = run_epoch(is_train=False, loader=tst_loader)
            mean_test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
            logger.info(
                f'Epoch {epoch + 1}/{config.epochs}, \
                    The training loss is {mean_test_epoch_loss:.3f}')
