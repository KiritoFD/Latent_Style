import json

import pytest
import torch
from PIL import Image

from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)


class TestKeyRenameMapper:
    @pytest.fixture()
    def dummy_batch(self):
        return {"image": 1, "text": 2, "label": "dummy_label"}

    @pytest.fixture()
    def mapper(self):
        return KeyRenameMapper(
            KeyRenameMapperConfig(
                key_map={"image": "image_tensor", "text": "text_tensor"}
            )
        )

    def test_mapper(self, mapper, dummy_batch):
        output_data = mapper(dummy_batch)
        assert output_data["image_tensor"] == 1
        assert output_data["text_tensor"] == 2
        assert output_data["label"] == "dummy_label"
        assert "image" not in output_data
        assert "text" not in output_data


class TestKeyRenameMapperWithCondition:
    @pytest.fixture(params=[1, 2])
    def dummy_batch(self, request):
        return {"image": 1, "text": 2, "label": request.param}

    @pytest.fixture(params=[{"image": "image_not_met", "text": "text_not_met"}, None])
    def else_key_map(self, request):
        return request.param

    @pytest.fixture()
    def mapper(self, else_key_map):
        return KeyRenameMapper(
            KeyRenameMapperConfig(
                key_map={"image": "image_tensor", "text": "text_tensor"},
                condition_key="label",
                condition_fn=lambda x: x == 1,
                else_key_map=else_key_map,
            )
        )

    def test_mapper(self, mapper, dummy_batch, else_key_map):
        output_data = mapper(dummy_batch)
        if dummy_batch["label"] == 1:
            assert output_data["image_tensor"] == 1
            assert output_data["text_tensor"] == 2
            assert output_data["label"] == 1
            assert "image" not in output_data
            assert "text" not in output_data
        elif else_key_map is not None:
            assert output_data["image_not_met"] == 1
            assert output_data["text_not_met"] == 2
            assert output_data["label"] == 2
            assert "image" not in output_data
            assert "text" not in output_data
        else:
            assert output_data["image"] == 1
            assert output_data["text"] == 2
            assert output_data["label"] == 2
            assert "image_tensor" not in output_data
            assert "text_tensor" not in output_data


class TestMapperWrapper:
    @pytest.fixture()
    def dummy_batch(self):
        return {"image": 1, "text": 2, "label": "dummy_label"}

    @pytest.fixture()
    def mapper(self):
        return MapperWrapper(
            mappers=[
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={"image": "image_tensor", "text": "text_tensor"}
                    )
                ),
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "image_tensor": "image_array",
                            "text_tensor": "text_array",
                        }
                    )
                ),
            ]
        )

    def test_mapper(self, mapper, dummy_batch):
        output_data = mapper(dummy_batch)
        assert output_data["image_array"] == 1
        assert output_data["text_array"] == 2
        assert output_data["label"] == "dummy_label"
        assert "image" not in output_data
        assert "text" not in output_data
        assert "image_tensor" not in output_data
        assert "text_tensor" not in output_data


class TestTorchvisionMapper:
    @pytest.fixture()
    def dummy_batch(self):
        return {
            "image": torch.randn(
                3,
                256,
                256,
            ),
            "text": 2,
            "label": "dummy_label",
        }

    @pytest.fixture()
    def mapper(self):
        return TorchvisionMapper(
            TorchvisionMapperConfig(
                key="image",
                transforms=["CenterCrop", "ToPILImage"],
                transforms_kwargs=[{"size": 224}, {}],
            )
        )

    def test_mapper(self, mapper, dummy_batch):
        output_data = mapper(dummy_batch)
        assert output_data["image"].size == (224, 224)
        assert isinstance(output_data["image"], Image.Image)
        assert output_data["text"] == 2
        assert output_data["label"] == "dummy_label"

    @pytest.fixture()
    def mapper_with_output_key(self):
        return TorchvisionMapper(
            TorchvisionMapperConfig(
                key="image",
                output_key="image_transformed",
                transforms=["CenterCrop", "ToPILImage"],
                transforms_kwargs=[{"size": 224}, {}],
            )
        )

    def test_mapper(self, mapper_with_output_key, dummy_batch):
        output_data = mapper_with_output_key(dummy_batch)
        assert output_data["image_transformed"].size == (224, 224)
        assert isinstance(output_data["image_transformed"], Image.Image)
        assert isinstance(output_data["image"], torch.Tensor)
        assert output_data["image"].size() == (3, 256, 256)
        assert output_data["text"] == 2
        assert output_data["label"] == "dummy_label"


class TestRescaleMapper:
    @pytest.fixture()
    def dummy_batch(self):
        return {
            "image": torch.rand(
                3,
                256,
                256,
            ),
            "text": 2,
            "label": "dummy_label",
        }

    @pytest.fixture()
    def mapper(self):
        return RescaleMapper(
            RescaleMapperConfig(
                input_key="image",
                output_key="image",
            )
        )

    def test_mapper(self, mapper, dummy_batch):
        output_data = mapper(dummy_batch)
        assert torch.all(output_data["image"] <= 1)
        assert torch.all(output_data["image"] >= -1)
        assert output_data["text"] == 2
        assert output_data["label"] == "dummy_label"
