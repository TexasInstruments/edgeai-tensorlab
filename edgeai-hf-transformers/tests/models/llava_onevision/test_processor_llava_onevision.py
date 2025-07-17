# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import shutil
import tempfile
import unittest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        LlavaOnevisionImageProcessor,
        LlavaOnevisionProcessor,
        LlavaOnevisionVideoProcessor,
        Qwen2TokenizerFast,
    )


@require_vision
class LlavaOnevisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaOnevisionProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = LlavaOnevisionImageProcessor()
        video_processor = LlavaOnevisionVideoProcessor()
        tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        processor_kwargs = self.prepare_processor_dict()

        processor = LlavaOnevisionProcessor(
            video_processor=video_processor, image_processor=image_processor, tokenizer=tokenizer, **processor_kwargs
        )
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def prepare_processor_dict(self):
        return {"chat_template": "dummy_template", "num_image_tokens": 6, "vision_feature_select_strategy": "default"}

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # chat_tempalate are tested as a separate test because they are saved in separate files
            if key != "chat_template":
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    # Copied from tests.models.llava.test_processor_llava.LlavaProcessorTest.test_chat_template_is_saved
    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
        expected_prompt = "<|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)
