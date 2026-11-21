"""Explicit headless experiment profile; existing viewer defaults stay unchanged."""
RULES={'hunger_per_round':.01,'healing_hp_per_round':1,'healing_hunger_limit':.1,
       'mate_seek_below':.1,'mate_cutoff':.3,'mating_damage_interrupt':True}


def current_rules(run,turn):
    run.set_hunger_rate(turn)
    run.world.healing_hp_per_round=1;run.world.healing_hunger_limit=.1;run.world.mate_hunger_limit=.3
    for policy in run.policies.values():
        policy.mate_hunger_threshold=.1;policy.resume_hunting=.3;policy.mate_hunger_limit=.3
