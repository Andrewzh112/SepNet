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
                 loss_type='addition'):
        self.config = config
        self.model = model
        self.trn_dataset = trn_dataset
        self.tst_dataset = tst_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.loss_type = loss_type

    def train(self,
              loss=torch.nn.MSELoss(),
              progress_interval=100,
              alpha=1.,
              sep_loss_anneal=None):
        config, device = self.config, self.device
        model = self.model.to(device)
        if self.loss_type == 'addition':
            criterion = AdditionLoss(loss=loss, alpha=alpha)
        elif self.loss_type == 'separation':
            criterion = SeperationLoss(loss=loss, alpha=alpha)
        else:
            raise InvalidLossError
        optimizer = torch.optim.Adam(
            model.parameters(), **config.opimizer_params
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
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
                optimizer.zero_grad()
                images = images.to(device)
                image_pairs, mixed_images, _ = mix_images(images)
                constructed_images = model(mixed_images)
                loss = criterion(constructed_images, image_pairs)
                losses.append(loss.item())
                if return_samples and samplex_idx == batch_idx:
                    # deep copy
                    sample_images = torch.tensor(constructed_images)
                    sample_truths = torch.cat(
                        [torch.stack(
                            [i0, i1],
                            dim=0) for i0, i1 in image_pairs],
                        dim=0
                    )
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()
                    scheduler.step()
            if return_samples:
                return losses, sample_images, sample_truths
            return losses

        @torch.no_grad()
        def save_progress(sample_images, sample_truths):
            images = unstack_images(
                sample_images,
                self.trn_dataset.statistics
            )
            sample_truths = unstack_images(
                sample_truths,
                self.trn_dataset.statistics,
                truths=True
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
            torchvision.utils.save_image(
                sample_truths,
                fp=os.path.join(
                    image_out_dir,
                    f'truths_epoch{epoch + 1}.jpg'
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
                    test_epoch_loss, sample_images, sample_truths = run_epoch(
                        is_train=False,
                        loader=tst_loader,
                        return_samples=True
                    )
                save_progress(sample_images, sample_truths)
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
            if sep_loss_anneal is not None and epoch > sep_loss_anneal:
                criterion.alpha = max(criterion.alpha*0.999, 0.5)
