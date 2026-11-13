import pytest
from experiments.approach_learning import collect_approach,normal_world
from experiments.approach_action import MOTOR_SCHEMA
from experiments.recurrent_agents import Agent,save,load
from spider_sim.dual_encounter import SPIDER,PLAYER


def test_new_corpus_tags_motor_contract_and_isolates_actor():
    data=collect_approach(Agent().eval(),SPIDER,700000,ordinary=1,focused=2)
    assert len(data)==3
    assert all(e["motor_schema"]==MOTOR_SCHEMA for e in data)
    assert [e["source"] for e in data]==["ordinary","focused","focused"]
    assert normal_world(PLAYER,700000,"ground").approach_actor==PLAYER


def test_new_checkpoint_contract_rejects_accidental_old_schema_loading(tmp_path):
    path=tmp_path/"approach.pt";save(Agent(),path,observation_schema=MOTOR_SCHEMA)
    with pytest.raises(ValueError,match="observation schema"):
        load(path)
    assert load(path,expected_schema=MOTOR_SCHEMA).kind=="gru"
