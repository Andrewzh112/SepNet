import torch
import torchvision
from torch.utils.data import DataLoader
from sepnet.utils import mix_images, unstack_images
from sepnet.loss import SeperationLoss, AdditionLoss
import numpy as np
from tqdm import tqdm
import os


class InvalidLossError(Exception):
    pass


class Trainer:
    def __init__(self,
                 model,
                 config,
                 trn_dataset,
                 tst_dataset,
                 loss_type='additon'):
        self.config = config
        self.model = model
        self.trn_dataset = trn_dataset
        self.tst_dataset = tst_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.loss_type = loss_type

    def train(self, progress_interval=100):
        config, device = self.config, self.device
        model = self.model.to(device)
        if self.loss_type == 'addition':
            criterion = AdditionLoss(loss=torch.nn.SmoothL1Loss())
        elif self.loss_type == 'separation':
            criterion = SeperationLoss(loss=torch.nn.SmoothL1Loss())
        else:
            raise InvalidLossError
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

        def run_epoch(is_train, loader, return_samples=False):
            losses = []
            if return_samples:
                samplex_idx = np.random.choice(list(range(len(loader))))
            for batch_idx, images in enumerate(loader):
                if isinstance(images, tuple):
                    images = images[0]
                images = images.to(device)
                image_pairs, mixed_images, _ = mix_images(images)
                constructed_images = model(mixed_images)
                loss = criterion(constructed_images, image_pairs)
                losses.append(loss.item())
                if return_samples and samplex_idx == batch_idx:
                    # deep copy
                    sample_images = torch.tensor(constructed_images)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()
                    scheduler.step()
            if return_samples:
                return losses, sample_images
            return losses

        @torch.no_grad()
        def save_progress(sample_images):
            images = unstack_images(
                sample_images,
                self.trn_dataset.statistics
            )
            image_out_dir = 'img_outputs'
            if not os.path.exists(image_out_dir):
                os.mkdir(image_out_dir)
            torchvision.utils.save_image(
                images,
                fp=os.path.join(
                    image_out_dir,
                    f'output_epoch{epoch + 1}.jpg'
                )
            )
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.model_path,
                    f'SepNet_Epoch{epoch + 1}.pt'
                )
            )

        for epoch in pbar:
            model.train()
            train_epoch_loss = run_epoch(is_train=True, loader=trn_loader)
            mean_train_epoch_loss = sum(
                train_epoch_loss) / len(train_epoch_loss)

            model.eval()
            if (epoch + 1) % progress_interval == 0:
                with torch.no_grad():
                    test_epoch_loss, sample_images = run_epoch(
                        is_train=False,
                        loader=tst_loader,
                        return_samples=True
                    )
                save_progress(sample_images)
            else:
                with torch.no_grad():
                    test_epoch_loss = run_epoch(
                        is_train=False, loader=tst_loader
                    )
            mean_test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
            tqdm.write(
                f'Epoch {epoch + 1}/{config.epochs}, \
                    Train loss: {mean_train_epoch_loss:.3f}, \
                        Test loss: {mean_test_epoch_loss:.3f}'
            )
