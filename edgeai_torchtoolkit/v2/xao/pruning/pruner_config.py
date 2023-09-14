

class PrunerType():
    NO_PRUNING = 0
    UNSTRUCTURED = 1
    N2M_PRUNING = 2
    CHANNEL_PRUNING = 3

    @classmethod
    def get_dict(cls):
        return {k:v for k,v in __class__.__dict__.items() if not k.startswith("__")}

    @classmethod
    def get_choices(cls):
        return {v:k for k,v in __class__.__dict__.items() if not k.startswith("__")}
