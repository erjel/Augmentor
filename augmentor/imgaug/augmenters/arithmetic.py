# Modified from https://github.com/aleju/imgaug/blob/0101108d4fed06bc5056c4a03e2bcb0216dac326/imgaug/augmenters/arithmetic.py
# Copyright (c) 2015 aleju
# See ../LICENCE for licensing information

from .. import parameters as iap
from . import meta

class AddElementwise(meta.Augmenter):
    """
    Add to the pixels of images values that are pixelwise randomly sampled.


    While the ``Add`` Augmenter samples one value to add *per image* (and
    optionally per channel), this augmenter samples different values per image
    and *per pixel* (and optionally per channel), i.e. intensities of
    neighbouring pixels may be increased/decreased by different amounts.


    **Supported dtypes**:


    See :func:`~imgaug.augmenters.arithmetic.add_elementwise`.


    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the pixels.


            * If an int, exactly that value will always be used.
            * If a tuple ``(a, b)``, then values from the discrete interval
              ``[a..b]`` will be sampled per image and pixel.
            * If a list of integers, a random value will be sampled from the
              list per image and pixel.
            * If a ``StochasticParameter``, then values will be sampled per
              image and pixel from that parameter.


    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).


    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.


    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.


    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.


    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.


    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AddElementwise(10)


    Always adds a value of 10 to all channels of all pixels of all input
    images.


    >>> aug = iaa.AddElementwise((-10, 10))


    Samples per image and pixel a value from the discrete interval
    ``[-10..10]`` and adds that value to the respective pixel.


    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)


    Samples per image, pixel *and also channel* a value from the discrete
    interval ``[-10..10]`` and adds it to the respective pixel's channel value.
    Therefore, added values may differ between channels of the same pixel.


    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)


    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.


    """


    def __init__(self, value=(-20, 20), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddElementwise, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


        self.value = iap.handle_continuous_param(
            value, "value", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")


    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch


        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])


        gen = enumerate(zip(images, per_channel_samples, rss[1:]))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            values = self.value.draw_samples(sample_shape, random_state=rs)


            batch.images[i] = add_elementwise(image, values)


        return batch


    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.per_channel]

class AdditiveGaussianNoise(AddElementwise):
    """
    Add noise sampled from gaussian distributions elementwise to images.


    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.


    **Supported dtypes**:


    See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.


    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the normal distribution from which the noise is sampled.


            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.


    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distribution that generates the noise.
        Must be ``>=0``. If ``0`` then `loc` will simply be added to all
        pixels.


            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.


    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).


    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.


    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.


    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.


    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.


    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255)


    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.


    >>> aug = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))


    Adds gaussian noise from the distribution ``N(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.


    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=True)


    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for red, green and blue channels of the same pixel).
    This leads to "colorful" noise.


    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=0.5)


    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.


    """
    def __init__(self, loc=0, scale=(0, 15), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        loc2 = iap.handle_continuous_param(
            loc, "loc", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        scale2 = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)


        value = iap.Normal(loc=loc2, scale=scale2)


        super(AdditiveGaussianNoise, self).__init__(
            value, per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)