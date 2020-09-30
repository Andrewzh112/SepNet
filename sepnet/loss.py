class AdditionLoss:
    def __init__(self, loss, alpha=0.99):
        self.loss = loss
        self.alpha = alpha

    def __call__(self, constructed_images, image_pairs):
        batch_size = constructed_images.size(0)
        addition_loss, part_loss = None, None
        for constructed_image, image_pair in zip(constructed_images, image_pairs):
            image0, image1 = image_pair
            rimage0, rimage1 = (constructed_image[:3, :, :],
                                constructed_image[3:, :, :])
            image0, image1 = image0.view(1, -1), image1.view(1, -1)
            rimage0, rimage1 = rimage0.view(1, -1), rimage1.view(1, -1)
            if addition_loss is None:
                addition_loss = self.loss(
                    0.5*image0 + 0.5*image1,
                    0.5*rimage0 + 0.5*rimage1
                )
            else:
                addition_loss += self.loss(
                    0.5*image0 + 0.5*image1,
                    0.5*rimage0 + 0.5*rimage1
                )
            if part_loss is None:
                part_loss = self.loss(rimage0, rimage1)
            else:
                part_loss += self.loss(rimage0, rimage1)
        return (self.alpha*addition_loss -
                (1-self.alpha)*part_loss) / batch_size


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
        pair_loss, part_loss = None, None
        for constructed_image, image_pair in zip(constructed_images, image_pairs):
            image0, image1 = image_pair
            rimage0, rimage1 = (constructed_image[:3, :, :],
                                constructed_image[3:, :, :])
            image0, image1 = image0.view(1, -1), image1.view(1, -1)
            rimage0, rimage1 = rimage0.view(1, -1), rimage1.view(1, -1)
            pair_loss0 = self._pair_loss(
                image0,
                image1,
                rimage0,
                rimage1
            )
            pair_loss1 = self._pair_loss(
                image0,
                image1,
                rimage1,
                rimage0
            )
            if pair_loss is None:
                pair_loss = min(pair_loss0, pair_loss1)
            else:
                pair_loss += min(pair_loss0, pair_loss1)
            if part_loss is None:
                part_loss = self.loss(rimage0, rimage1)
            else:
                part_loss += self.loss(rimage0, rimage1)
        separation_loss = (self.alpha*pair_loss -
                           (1-self.alpha)*part_loss) / batch_size
        return separation_loss
