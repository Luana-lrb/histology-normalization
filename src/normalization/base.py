class BaseNormalizer:
    def __init__(self, reference_image):
        self._fit(reference_image)

    def _fit(self, reference_image):
        raise NotImplementedError

    def __call__(self, image_path):
        raise NotImplementedError
