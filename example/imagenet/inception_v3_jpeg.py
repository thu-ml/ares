from realsafe.defense.jpeg_compression import jpeg_compression

import inception_v3


def load(session):
    model = InceptionV3Jpeg()
    model.load(session)
    return model


@jpeg_compression(quality=75)
class InceptionV3Jpeg(inception_v3.InceptionV3):
    pass
