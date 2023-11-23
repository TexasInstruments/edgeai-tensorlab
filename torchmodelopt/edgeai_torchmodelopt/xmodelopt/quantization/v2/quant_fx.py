from .quant_fx_base import QuantFxBaseModule


class QATFxModule(QuantFxBaseModule):
    def __init__(self, *args, backend='qnnpack', **kwargs):
        super().__init__(*args, is_qat=True, backend=backend, **kwargs)


class PTQFxModule(QuantFxBaseModule):
    def __init__(self, *args, backend='qnnpack', **kwargs):
        super().__init__(*args, is_qat=False, backend=backend, **kwargs)

