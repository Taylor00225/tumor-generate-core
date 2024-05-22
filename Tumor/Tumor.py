import random
from typing import Any, Mapping, Hashable, Dict

import numpy as np
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.transforms.transform import RandomizableTransform

from .utils import SynthesisTumor


class TumorGenerated(MapTransform, RandomizableTransform):
    tumor_prob: list | tuple | np.ndarray
    textures: Any
    sigma_a: float = 3.0
    sigma_b: float = 0.6
    predefined_texture_shape: tuple
    texture: Any

    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 1.0,
                 tumor_prob: list | tuple | np.ndarray = (0.2, 0.2, 0.2, 0.2, 0.2),
                 allow_missing_keys: bool = False
                 ) -> None:
        """

        :param keys:
        :param prob: probability of whether generate tumor or not
        :param tumor_prob: tumor types probability
        :param allow_missing_keys:
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        random.seed(0)
        np.random.seed(0)

        self.b = 15

        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)
        # self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']
        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'medium']

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        sigma_a = np.std(d['image'][0])
        # if self._do_transform and (np.max(d['label']) <= 1):
        if self._do_transform:
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            print(f"Tumor_type : {tumor_type}")
            print("Starting!")
            d['image'][0], d['label'][0] = SynthesisTumor(d['image'][0], d['label'][0], tumor_type, self.b)
            print("Finish!")
        return d
