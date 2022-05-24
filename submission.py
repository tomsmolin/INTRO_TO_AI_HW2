from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random

# TODO: ask in piazze it's ok to import the module
import math
import time

class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.heuristic(child, agent_id) for child in children]
        # # Tom - DEBUG
        # print("######################")
        # print(children_heuristics)
        # print("######################")
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

        # Prioritize passenger pickup/drop-off
        #index_selected = children_heuristics[::-1].index(max_heuristic)
        # return operators[len(children_heuristics) - index_selected - 1]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        dist_from_passengers = [manhattan_distance(p.position, taxi.position) for p in env.passengers]

        if taxi.passenger is None:
            compensation = [manhattan_distance(p.position, p.destination) for p in env.passengers]
            worth = [0 for _ in range(len(env.passengers))]
            for idx in range(len(env.passengers)):
                # The next cond. is placed to make sure the agent doesn't prioritize
                # picking up a passenger with 0 reward - as defined in the heuristic
                if compensation[idx] == 0:
                    worth[idx] = 0
                else:
                    worth[idx] = 12 + 12*taxi.cash - (dist_from_passengers[idx] + compensation[idx])
            max_worth = max(worth)
            return (max_worth)

        else:   # There is a passenger on the taxi
            passenger = taxi.passenger
            compensation = manhattan_distance(passenger.destination, passenger.position)
            # The next cond. is placed to make sure the agent doesn't prioritize
            # picking up a passenger with 0 reward - as defined in the heuristic
            if compensation == 0:
                return 0
            else:
                return (12 + 12*taxi.cash - manhattan_distance(taxi.position, passenger.destination))


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        max_depth = 4

        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        best_op_so_far = operators[0]
        # Should think and inquire about the time limits here more
        while (time.time() - start) < (0.25 * time_limit):
            children_values = [self.rb_minimax(child, agent_id, False, max_depth-1) for child in children]
            children_max = max(children_values)
            # Prioritize passenger pickup/drop-off
            index_selected = children_values[::-1].index(children_max)
            best_op_so_far = operators[len(children_values) - index_selected - 1]
            max_depth += 1
        # TOM DEBUG
        # print(f'agent {agent_id} max depth: {max_depth}')
        return best_op_so_far

    def rb_minimax(self, env, agent_id, our_turn, depth):
        if env.done() or depth == 0:
            return self.heuristic(env, agent_id)

        # Turn <- Turn(State)
        agent_to_play = agent_id if our_turn else (agent_id+1) % 2
        # Children <- Succ(State)
        operators = env.get_legal_operators(agent_to_play)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_to_play, op)

        if our_turn:
            curr_max = -math.inf
            for child in children:
                value = self.rb_minimax(child, agent_id, False, depth-1)
                curr_max = max(curr_max, value)
            return curr_max

        else:
            curr_min = math.inf
            for child in children:
                value = self.rb_minimax(child, agent_id, True, depth-1)
                curr_min = min(curr_min, value)
            return curr_min

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id+1) % 2)
        if env.done():
            return taxi.cash - other_taxi.cash

        dist_from_passengers = [manhattan_distance(p.position, taxi.position) for p in env.passengers]

        if taxi.passenger is None:
            compensation = [manhattan_distance(p.position, p.destination) for p in env.passengers]
            worth = [0 for _ in range(len(env.passengers))]
            for idx in range(len(env.passengers)):
                # The next cond. is placed to make sure the agent doesn't prioritize
                # picking up a passenger with 0 reward - as defined in the heuristic
                if compensation[idx] == 0:
                    worth[idx] = 0
                else:
                    worth[idx] = 12 + 12*taxi.cash - (dist_from_passengers[idx] + compensation[idx])
            max_worth = max(worth)
            return (max_worth)

        else:   # There is a passenger on the taxi
            passenger = taxi.passenger
            compensation = manhattan_distance(passenger.destination, passenger.position)
            # The next cond. is placed to make sure the agent doesn't prioritize
            # picking up a passenger with 0 reward - as defined in the heuristic
            if compensation == 0:
                return 0
            else:
                return (12 + 12*taxi.cash - manhattan_distance(taxi.position, passenger.destination))

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        max_depth = 4

        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        best_op_so_far = operators[0]
        # Should think and inquire about the time limits here more
        while (time.time() - start) < (0.25 * time_limit):
            children_values = [self.rb_alpha_beta(child, agent_id, False, max_depth-1, -math.inf, math.inf)
                               for child in children]
            children_max = max(children_values)
            # Prioritize passenger pickup/drop-off
            index_selected = children_values[::-1].index(children_max)
            best_op_so_far = operators[len(children_values) - index_selected - 1]
            max_depth += 1
        # TOM DEBUG
        # print(f'agent {agent_id} max depth: {max_depth}')
        return best_op_so_far
        # raise NotImplementedError()

    def rb_alpha_beta(self, env, agent_id, our_turn, depth, alpha, beta):
        if env.done() or depth == 0:
            return self.heuristic(env, agent_id)

        # Turn <- Turn(State)
        agent_to_play = agent_id if our_turn else (agent_id+1) % 2
        # Children <- Succ(State)
        operators = env.get_legal_operators(agent_to_play)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_to_play, op)
        # children.reverse() - an idea to improve alpha-beta comapring to

        if our_turn:
            curr_max = -math.inf
            for child in children:
                value = self.rb_alpha_beta(child, agent_id, False, depth-1, alpha, beta)
                curr_max = max(curr_max, value)
                alpha = max(curr_max, alpha)
                # A lot of equal children - may be misleading
                # Prone only whey greater than ------> NOT HAPPENING AT THE MOMENT AS IT DOESN'T MATCH THE ORIGINAL ALG
                if curr_max >= beta:
                    return math.inf

            return curr_max

        else:
            curr_min = math.inf
            for child in children:
                value = self.rb_alpha_beta(child, agent_id, True, depth-1, alpha, beta)
                curr_min = min(curr_min, value)
                beta = min(curr_min, beta)
                # A lot of equal children - may be misleading
                # Prone only whey greater than ------> NOT HAPPENING AT THE MOMENT AS IT DOESN'T MATCH THE ORIGINAL ALG
                if curr_min <= alpha:
                    return (-math.inf)

            return curr_min

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id+1) % 2)
        if env.done():
            return taxi.cash - other_taxi.cash

        dist_from_passengers = [manhattan_distance(p.position, taxi.position) for p in env.passengers]

        if taxi.passenger is None:
            compensation = [manhattan_distance(p.position, p.destination) for p in env.passengers]
            worth = [0 for _ in range(len(env.passengers))]
            for idx in range(len(env.passengers)):
                # The next cond. is placed to make sure the agent doesn't prioritize
                # picking up a passenger with 0 reward - as defined in the heuristic
                if compensation[idx] == 0:
                    worth[idx] = 0
                else:
                    worth[idx] = 12 + 12*taxi.cash - (dist_from_passengers[idx] + compensation[idx])
            max_worth = max(worth)
            return (max_worth)

        else:   # There is a passenger on the taxi
            passenger = taxi.passenger
            compensation = manhattan_distance(passenger.destination, passenger.position)
            # The next cond. is placed to make sure the agent doesn't prioritize
            # picking up a passenger with 0 reward - as defined in the heuristic
            if compensation == 0:
                return 0
            else:
                return (12 + 12*taxi.cash - manhattan_distance(taxi.position, passenger.destination))


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
