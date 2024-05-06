import os
import shutil
import sys
import tempfile

import pytest
import torch.hub as hub


def sum_of_model_parameters(model):
    s = 0
    for p in model.parameters():
        s += p.sum()
    return s


SUM_OF_PRETRAINED_RESNET18_PARAMS = -12703.9931640625


@pytest.mark.skipif("torchvision" in sys.modules, reason="TestHub must start without torchvision imported")
class TestHub:
    # Only run this check ONCE before all tests start.
    # - If torchvision is imported before all tests start, e.g. we might find _C.so
    #   which doesn't exist in downloaded zip but in the installed wheel.
    # - After the first test is run, torchvision is already in sys.modules due to
    #   Python cache as we run all hub tests in the same python process.

    def test_load_from_github(self):
        hub_model = hub.load("pytorch/vision", "resnet18", weights="DEFAULT", progress=False)
        assert sum_of_model_parameters(hub_model).item() == pytest.approx(SUM_OF_PRETRAINED_RESNET18_PARAMS)

    def test_set_dir(self):
        temp_dir = tempfile.gettempdir()
        hub.set_dir(temp_dir)
        hub_model = hub.load("pytorch/vision", "resnet18", weights="DEFAULT", progress=False)
        assert sum_of_model_parameters(hub_model).item() == pytest.approx(SUM_OF_PRETRAINED_RESNET18_PARAMS)
        assert os.path.exists(temp_dir + "/pytorch_vision_master")
        shutil.rmtree(temp_dir + "/pytorch_vision_master")

    def test_list_entrypoints(self):
        entry_lists = hub.list("pytorch/vision", force_reload=True)
        assert "resnet18" in entry_lists


if __name__ == "__main__":
    pytest.main([__file__])
