from .PGD import PGD


class BIM(PGD):
    def __init__(self, *args, **kwargs):
        kwargs["random_start"] = False
        super(BIM, self).__init__(*args, **kwargs)
