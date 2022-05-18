from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random

# TODO: piazza - is it ok to import these packages
import math


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.heuristic(child, agent_id) for child in children]
        # # Tom - DEBUG
        print("######################")
        print(children_heuristics)
        print("######################")
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        #pass
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id+1) % 2)
        dist_from_passengers = [manhattan_distance(p.position, taxi.position) for p in env.passengers]

        if taxi.passenger is None:
            compensation = [manhattan_distance(p.position, p.destination) for p in env.passengers]
            worth = [0 for _ in range(len(env.passengers))]
            for idx in range(len(env.passengers)):
                # The next cond. is placed to make sure the agent doesn't prioritize
                # picking up a passenger with 0 reward - as defined in the heuristic
                if compensation[idx] == 0:
                    worth[idx] = -math.inf
                    #worth[idx] = (-1) * 10000
                else:
                    worth[idx] = 12 + 10*(taxi.cash - other_taxi.cash) \
                                    + (taxi.fuel - other_taxi.fuel) \
                                    - (dist_from_passengers[idx] + compensation[idx])
            max_worth = max(worth)
            return (max_worth)

        else:   # There is a passenger on the taxi
            passenger = taxi.passenger
            compensation = manhattan_distance(passenger.destination, passenger.position)
            # The next cond. is placed to make sure the agent doesn't prioritize
            # picking up a passenger with 0 reward - as defined in the heuristic
            if compensation == 0:
                return -math.inf
                # return (-1) * 10000
            else:
                return (12 + 10*(taxi.cash - other_taxi.cash) \
                           + (taxi.fuel - other_taxi.fuel) \
                           - manhattan_distance(taxi.position, passenger.destination))


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
