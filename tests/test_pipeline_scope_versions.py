"""Persisted recipes and values across the Pipeline branch-scope change."""

import numpy as np
import pynbody
import pytest
from test_pipeline_branch_scopes import branch_snapshot

import pynbodyext as pe
from pynbodyext.calculate import CalculatorBase, Pipeline, PropertyBase
from pynbodyext.core.calculate.result.signature import CalculatorSignature, calculator_from_signature
from pynbodyext.filters import Disc
from pynbodyext.properties import ParamSum
from pynbodyext.transforms import AlignVec


class FieldSummary(Pipeline):
    """A project recipe with its own constructor, signature version, and rebuild hook."""

    signature_name = "test_pipeline_scope_versions.FieldSummary"
    signature_version = 7

    def __init__(self, field):
        super().__init__({"total": ParamSum(field)})
        self.field = field

    def signature_payload(self):
        return {"field": self.field}

    @classmethod
    def from_signature_payload(cls, payload):
        return cls(payload["field"])


# Addresses recorded at c56120f, before named outputs isolated their transforms.
@pytest.mark.parametrize(
    ("recipe", "legacy_hash"),
    [
        (Pipeline({"mass": ParamSum("mass")}), "193537dae943cf0da09819130e73c53a4def8b7fbe79e484019d77f4b7a6031d"),
        (FieldSummary("mass"), "dc744a66be253fa92f15b751ec650f2200ee76801ed5f4a63e77b388a926f71b"),
        (ParamSum("mass") & FieldSummary("mass"), "5b72d5fe09b5809e6a2355859311fa31ea4fe38129ab2cf8533775374b074f86"),
        (
            Pipeline({"summary": FieldSummary("mass")}),
            "6bdfceb627f0d624af840be42c355e741579e50b6ad0acc5726e2365f53a05ff",
        ),
    ],
    ids=["pipeline", "custom-constructor", "combined", "nested"],
)
def test_branch_recipes_round_trip_under_new_addresses(recipe, legacy_hash):
    signature = recipe.to_signature()
    assert signature.hash != legacy_hash
    assert calculator_from_signature(signature.to_json()).to_signature() == signature
    if isinstance(recipe, FieldSummary):
        assert signature.payload["version"] == 7


def test_standalone_property_retains_its_persistent_address():
    sig = ParamSum("mass").to_signature()
    assert sig.payload == {
        "class": "pynbodyext.properties.ParamSum",
        "node": "dataclass",
        "init": {"parameter": "mass"},
    }
    assert sig.hash == "846c60c4f63348210becf3a058d8d5258fdfe6384e93c332979c793fc148f839"


@PropertyBase.dataclass
class Membership(PropertyBase):
    signature_name = "test_pipeline_scope_versions.Membership"

    def calculate(self, sim, params=None):
        return sim.get_index_list(sim.ancestor).copy()


class LegacyPipeline(CalculatorBase):
    """The c56120f success path and its serializer, to seed a real database."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def declared_dependencies(self):
        return list(self.outputs.values())

    def execute(self, ctx, input):
        values = {}
        for key, child in self.outputs.items():
            node = ctx.evaluate(child, input)
            values[key] = ctx.nodes.runtime[node.node_id].public_value
            ctx.nodes.named[key] = node.node_id
        return values

    def to_signature(self, *, inline_array_bytes=128):
        return CalculatorSignature(
            payload={
                "node": "pipeline",
                "class": "pynbodyext.core.calculate.Pipeline",
                "outputs": [
                    {"key": key, "value": child.to_signature(inline_array_bytes=inline_array_bytes).payload}
                    for key, child in self.outputs.items()
                ],
            }
        )


def membership_pipeline(pipeline_type):
    disc = Disc(3.0, 0.1)
    rotated = Membership().transform(AlignVec((1.0, 0.0, 0.0))).filter(disc)
    return pipeline_type({"rotated": rotated, "native": Membership().filter(disc)})


def test_upgrade_recomputes_auto_and_preserves_frozen_study_cells(tmp_path):
    path, storage = tmp_path / "snapshot", tmp_path / "results.sqlite"
    sim = branch_snapshot()
    sim["mass"][:] = np.arange(1.0, 7.0)
    sim.properties["time"] = 0.0
    (tmp_path / "units.param").write_text("dKpcUnit = 1\ndMsolUnit = 1\nbComove = 0\n")
    sim.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(path))

    with pe.Database(storage) as db:
        source = db.register(pynbody.load(str(path)), simulation="box", snapshot="99")
        mass = ParamSum("mass").run(source).store_ref
        legacy = membership_pipeline(LegacyPipeline)
        signature = legacy.to_signature()
        assert signature.hash == "79310109dd270fc7a6f307ccac50f69eb664e6689a3eab57df28d2c5b2c346a5"
        release = legacy.run(source)
        for key in ("rotated", "native"):
            np.testing.assert_array_equal(release.value[key], [3, 4, 5])
        study = db.study(objects=[source], calculations={"members": signature})
        study.save(name="as-published")
        pinned = study.collect().cell(source, "members").ref
        assert pinned.record_ref == release.store_ref

    with pe.Database(storage) as db:
        source = db.snapshots[source.key]
        recipe = membership_pipeline(Pipeline)
        fresh = recipe.run(source, mode="auto")
        assert not fresh.from_store
        np.testing.assert_array_equal(fresh.value["native"], [2, 3, 5])
        np.testing.assert_array_equal(fresh.value["rotated"], [3, 4, 5])
        assert fresh.store_ref.cache_key != pinned.cache_key
        hit = recipe.run(source, mode="auto")
        assert hit.from_store and hit.store_ref == fresh.store_ref
        refreshed = recipe.run(source, mode="refresh")
        assert db.versions(fresh.store_ref) == [fresh.store_ref, refreshed.store_ref]

        standalone = ParamSum("mass").run(source)
        assert standalone.from_store and standalone.store_ref == mass
        historical = db.studies["as-published"]
        cell = historical.collect().cell(source, "members")
        assert cell.ref == pinned
        np.testing.assert_array_equal(cell.value["native"], [3, 4, 5])
        np.testing.assert_array_equal(db.fetch(pinned).value["native"], [3, 4, 5])
        assert historical.run().cell(source, "members").from_store
        with pytest.raises(ValueError, match="frozen signature"):
            historical.run(mode="refresh")
        assert db.versions(pinned) == [pinned.record_ref]
