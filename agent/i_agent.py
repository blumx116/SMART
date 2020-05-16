from typing import Generic

from misc.typevars import Environment, State,Action,  Reward, Goal 

class IAgent(Generic[Environment, State, Action, Reward, Goal]):
    def reset(self, env: Environment, state: State, goal: Goal) -> None:
         pass 

    def act(self, state: State, goal: Goal) -> Action:
        pass 

    def view(self, state: State, action: Action, reward: Reward) -> None:
        pass 

    def optimize(self) -> None:
        pass