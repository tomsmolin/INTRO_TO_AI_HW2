from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random

import math
import time


# WAS a Method in alpha-beta and minimax Agents
# now an outside function to avoid duplication
def heuristic_function(env: TaxiEnv, taxi_id: int):
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



class AgentGreedyImproved(AgentGreedy):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [heuristic_function(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentMinimax(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        max_depth = 1

        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        best_op_so_far = operators[0]
        while (time.time() - start) < (0.1 * time_limit):
            children_values = [self.rb_minimax(child, agent_id, False, max_depth-1) for child in children]
            children_max = max(children_values)
            # Prioritize passenger pickup/drop-off
            index_selected = children_values[::-1].index(children_max)
            best_op_so_far = operators[len(children_values) - index_selected - 1]
            max_depth += 1
        return best_op_so_far

    def rb_minimax(self, env, agent_id, our_turn, depth):
        if env.done() or depth == 0:
            if env.done():
                taxi = env.get_taxi(agent_id)
                other_taxi = env.get_taxi((agent_id+1) % 2)
                return taxi.cash - other_taxi.cash

            return heuristic_function(env, agent_id)

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


class AgentAlphaBeta(Agent):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        max_depth = 1

        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        best_op_so_far = operators[0]
        while (time.time() - start) < (0.05 * time_limit):
            children_values = [self.rb_alpha_beta(child, agent_id, False, max_depth-1, -math.inf, math.inf)
                               for child in children]
            children_max = max(children_values)
            # Prioritize passenger pickup/drop-off
            index_selected = children_values[::-1].index(children_max)
            best_op_so_far = operators[len(children_values) - index_selected - 1]
            max_depth += 1
        return best_op_so_far

    def rb_alpha_beta(self, env, agent_id, our_turn, depth, alpha, beta):
        if env.done() or depth == 0:
            if env.done():
                taxi = env.get_taxi(agent_id)
                other_taxi = env.get_taxi((agent_id+1) % 2)
                return taxi.cash - other_taxi.cash
            return heuristic_function(env, agent_id)

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
                value = self.rb_alpha_beta(child, agent_id, False, depth-1, alpha, beta)
                curr_max = max(curr_max, value)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max

        else:
            curr_min = math.inf
            for child in children:
                value = self.rb_alpha_beta(child, agent_id, True, depth-1, alpha, beta)
                curr_min = min(curr_min, value)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return (-math.inf)
            return curr_min



class AgentExpectimax(Agent):
        
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        max_depth = 1
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        best_op_so_far = operators[0]
        while (time.time() - start) < (0.07 * time_limit):
            children_values = [self.rb_expectimax(child, agent_id, False, max_depth-1, -math.inf, math.inf)
                               for child in children]
            children_max = max(children_values)
            # Prioritize passenger pickup/drop-off
            index_selected = children_values[::-1].index(children_max)
            best_op_so_far = operators[len(children_values) - index_selected - 1]
            max_depth += 1
        return best_op_so_far

    # maybe add alpha? the opponent is random so beta is less of an option, but perhaps alpha?
    def rb_expectimax(self, env, agent_id, our_turn, depth, alpha = -math.inf, beta = math.inf):
        if env.done() or depth == 0:
            if env.done():
                taxi = env.get_taxi(agent_id)
                other_taxi = env.get_taxi((agent_id+1) % 2)
                return taxi.cash - other_taxi.cash

            return heuristic_function(env, agent_id)

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
                value = self.rb_expectimax(child, agent_id, False, depth-1)
                curr_max = max(curr_max, value)
            return curr_max
        else:
            # treat the opponent as a random greedy player
            #check if the special operators (pickup, dropoff, fillup are in the operators)      
            num = 0
            for op in operators:
                if op == 'pick up passenger' or op == 'drop off passenger' or op == 'refuel' or op == 'park':
                    num += 2
                else:
                    num += 1
            pval = 1/num
            values =[] 
            total = 0
            for child in children:
                values.append(self.rb_expectimax(child, agent_id, True, depth-1))
            for val, op in zip(values, operators):
                if op == 'pick up passenger' or op == 'drop off passenger' or op == 'refuel' or op == 'park':
                    total += (2*pval*val)
                else:
                    total += (1*pval*val)
            return total
