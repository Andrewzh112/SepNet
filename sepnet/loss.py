class SeperationLoss:
    def __init__(self, loss, alpha=0.8):
        self.loss = loss
        self.alpha = alpha

    def _pair_loss(self, image0, image1, rimage0, rimage1):
        loss0 = self.loss(image0, rimage0)
        loss1 = self.loss(image1, rimage1)
        return loss0 + loss1

    def __call__(self, constructed_images, image_pairs):
        batch_size = constructed_images.size(0)
        pair_loss, separation_loss = None, None
        for constructed_image, image_pair in zip(constructed_images, image_pairs):
            image0, image1 = image_pair
            rimage0, rimage1 = (constructed_image[:3, :, :],
                                constructed_image[3:, :, :])
            image0, image1 = image0.view(1, -1), image1.view(1, -1)
            rimage0, rimage1 = rimage0.view(1, -1), rimage1.view(1, -1)
            pair_loss0 = self._pair_loss(
                image0.detach(),
                image1.detach(),
                rimage0.detach(),
                rimage1.detach()
            )
            pair_loss1 = self._pair_loss(
                image0.detach(),
                image1.detach(),
                rimage1.detach(),
                rimage0.detach()
            )
            if pair_loss0 < pair_loss1:
                if pair_loss is None:
                    pair_loss = self._pair_loss(
                        image0, image1, rimage0, rimage1
                    )
                else:
                    pair_loss += self._pair_loss(
                        image0, image1, rimage0, rimage1
                    )
            else:
                if pair_loss is None:
                    pair_loss = self._pair_loss(
                        image0, image1, rimage1, rimage0
                    )
                else:
                    pair_loss += self._pair_loss(
                        image0, image1, rimage1, rimage0
                    )
            if separation_loss is None:
                separation_loss = self.loss(rimage0, rimage1)
            else:
                separation_loss += self.loss(rimage0, rimage1)
        separation_loss = (self.alpha*pair_loss -
                           (1-self.alpha)*separation_loss) / batch_size
        return separation_loss
