"""Select a currently perceived meal once, before activity inference.

A remembered private track is a preference, never a hidden position. Both the
actor and executor receive the same selected observation for this decision.
"""


def select_meal(observation,remembered_track=None):
    meals=observation.get('food_in_view',[])
    return (next((meal for meal in meals if meal['track']==remembered_track),None)
            or min(meals,key=lambda meal:meal['range_ft'],default=None))
