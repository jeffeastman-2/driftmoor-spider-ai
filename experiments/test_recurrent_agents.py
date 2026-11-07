import torch
from experiments.recurrent_agents import Agent,save,load
from experiments.run_recurrent_agents import batch,objective

def test_sessions_have_private_memory_and_reset():
    torch.manual_seed(1)
    agent=Agent().eval()
    a,b=agent.session(),agent.session()
    first=a.choose([1.]*41)
    assert b.hidden is None
    assert a.hidden is not None and not a.hidden.requires_grad
    assert first==b.choose([1.]*41)
    a.choose([0.]*41)
    assert not torch.equal(a.hidden,b.hidden)
    assert first==agent.session().choose([1.]*41)

def test_save_load_and_memoryless_shape(tmp_path):
    for kind in ("gru","mlp"):
        agent=Agent(kind).eval()
        path=tmp_path/(kind+".pt")
        save(agent,path)
        assert agent.session().choose([0.]*41)==load(path).session().choose([0.]*41)

def test_padding_does_not_contribute_to_loss():
    row={"observation":[0.]*41,"memory_target":[0.]*5,"next_observation":[0.]*41,
         "feedback":[0.,0.],"teacher_action":0,"executed_action":0,"contact_missing":True}
    data=batch([{"actors":{"a":[row]}},{"actors":{"a":[row,row]}}],"a")
    agent=Agent()
    loss=objective(agent,data)
    changed=list(data)
    changed[2]=data[2].clone()
    changed[2][~data[3]]=100000
    assert torch.equal(loss,objective(agent,changed))
    loss.backward()
    assert agent.encoder[0].weight.grad is not None
