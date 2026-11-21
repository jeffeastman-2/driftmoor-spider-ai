"""Teacher labels on a student's private history, never an engine controller.

The advisor mirrors the student's prior executed goal/target bookkeeping. It
maintains only a hunger-derived search latch of its own, not an imagined world
trajectory or alternative motor memory. Its proposals are discarded.
"""
from types import SimpleNamespace
from spider_sim.feeding import FeedingPolicy


class LabelOnlyHunter:
    def __init__(self):
        self.locomotion=None;self.last_plan={}
        self.task=SimpleNamespace(previous=None,track=None,last=None)

    def choose(self,obs):return {'action':'wait'}


class RecoveryAdvisor:
    def __init__(self):self.teacher=FeedingPolicy(LabelOnlyHunter())

    def recommend(self,observation,student):
        teacher=self.teacher
        # These describe this actor's actual previous activity and motor memory,
        # not hidden information about an opponent or the engine's future.
        for key in ('goal','meal','mate_track','defense_track','defense_age',
                    'mate_hunger_threshold','resume_hunting','mate_hunger_limit','courtship_ready'):
            if hasattr(student,key):setattr(teacher,key,getattr(student,key))
        command=teacher.choose(observation)
        return teacher.goal,command
