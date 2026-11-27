import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from superagent import table_foundation as table
from superagent.ab_methods import ALIGNED
from superagent.ab_run import registered_capabilities
from tests.test_ab_scheduler import recorded


class OptionalTableModelTests(TestCase):
    def test_optional_cache_is_pinned_and_only_available_if_identical(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ,{table.WEIGHT_ENV:""},clear=True):
            path=Path(tmp)/"public-model.ckpt"
            path.write_bytes(b"tiny incorrect placeholder")
            with patch.object(table,"find_spec",return_value=True), patch.object(table,"version",return_value="2.0.3"), \
                 patch("huggingface_hub.try_to_load_from_cache",return_value=str(path)) as cache:
                self.assertIsNone(table.checkpoint())
                cache.assert_called_once_with(table.MODEL,table.FILENAME,revision=table.REVISION,cache_dir=None)

    def test_native_components_respect_distinct_binding_restrictions(self):
        spec=dict(recorded()["task"],kind="categorical_tabular",input_schema={})
        spec["metric"]=dict(spec["metric"],labels=[0,1])
        with patch.object(table,"checkpoint",return_value=Path("verified")):
            possible=registered_capabilities(spec,800,repeated_context=True,comparable_context=True)
            self.assertEqual(possible[table.NAME]["classifier_fits"],1)
            self.assertIn(ALIGNED,possible)
            self.assertNotIn(ALIGNED,registered_capabilities(spec,800,repeated_context=True))
            for flag in ("no_external_data","no_pretrained_assets","linear_only"):
                restricted=registered_capabilities(dict(spec,method_restrictions=[flag]),800,
                                                   repeated_context=True,comparable_context=True)
                self.assertNotIn(table.NAME,restricted)
                self.assertEqual(ALIGNED in restricted,flag!="linear_only")
