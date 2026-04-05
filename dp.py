'''
This house the dynamic programming algorithm to solve the problem. such as policy iteration, value iteration
'''
import numpy as np
import matplotlib.pyplot as plt

# the policy iteration alg
# get the card probability -> 10 is for 10, queen, prince and king so it is 4 from 14 other wise each card has prob of 1 from 13
def card_prob(card):
    return 4/13 if card == 10 else 1/13

# the total probability of the dealer where the dealer stand if the total is between 17 and 21 and hit if smaller tahn 17
def dealer_probs(total, memo_dealer):
    if total in memo_dealer:
        return memo_dealer[total], memo_dealer
    if total >= 17:
        if total > 21:
            return {"bust": 1.0}, memo_dealer
        else:
            return {total: 1.0}, memo_dealer
    
    outcomes = {}
    for card in range(1, 11):
        p = card_prob(card)
        new_total = total + card
        sub_probs, memo_dealer = dealer_probs(new_total, memo_dealer)
        for result, prob in sub_probs.items():
            outcomes[result] = outcomes.get(result, 0) + p * prob
    
    memo_dealer[total] = outcomes
    return outcomes, memo_dealer

# the prob of winning/losing/drew if stand in chosen
def stand_value(player_total, dealer_card, gamma, memo_dealer):
    outcomes, memo_dealer = dealer_probs(dealer_card, memo_dealer)
    val = 0
    for outcome, prob in outcomes.items():
        if outcome == "bust" or player_total > outcome:
            val += prob * 1 * gamma
        elif player_total < outcome:
            val += prob * -1 * gamma
        else:  # draw
            val += prob * 0 * gamma
    return val, memo_dealer

# the prob of winning/losing/drew if hit is chosen
def hit_value(V, player_total, dealer_card, gamma):
    val = 0
    for card in range(1, 11):
        p = card_prob(card)
        new_total = player_total + card
        if new_total > 21:
            val += p * -1 * gamma
        else:
            val += p * V.get((new_total, dealer_card, False),0) * gamma
    return val
def policy_evaluation (V, states, policy, gamma, theta, memo_dealer):
    while True:
        delta = 0.0
        for s in states:
            player, dealer, ace = s
            if player > 21:
                continue
            v = V[s]
            action = policy[s]
            if action == 0:  # stand
                V[s], memo_dealer = stand_value(player, dealer, gamma, memo_dealer)
            else:  # hit
                V[s] = hit_value(V, player, dealer, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, memo_dealer

def policy_improvement (V, states, policy, gamma, memo_dealer):
    is_policy_stable = True
    for s in states:
        player, dealer, ace = s
        if player > 21:
            continue
        old_action = policy[s]
        hit = hit_value(V, player, dealer, gamma)
        stick, memo_dealer = stand_value(player, dealer, gamma, memo_dealer)
        best_action = 1 if hit > stick else 0
        policy[s] = best_action
        if old_action != best_action:
            is_policy_stable = False
    return is_policy_stable, policy, memo_dealer

# itteration to get which has the best prob hit/stand
def policy_itteration (gamma=1.0, theta=0.001):
    # all initial states
    states = [(player, dealer, usable_ace)
            for player in range(4, 22)
            for dealer in range(1, 11)
            for usable_ace in [True, False]]
    
    V = {s: 0 for s in states}  # value function
    policy = {s: 0 for s in states}  # initial policy (stand by default)
    is_policy_stable = False
    iteration = 0

    memo_dealer = {}

    while not is_policy_stable:
        iteration += 1
        # Policy evaluation 
        V, memo_dealer = policy_evaluation(V, states, policy, gamma, theta, memo_dealer)
        
        # Policy improvement
        is_policy_stable, policy, memo_dealer = policy_improvement(V, states, policy, gamma, memo_dealer)
    return policy, V

# # example
# sample_state = (11, 8, False)
# policy = policy_itteration()
# print(f"Best action for state: {'HIT' if policy[sample_state]==1 else 'STAND'}")

# value iterations
# itteration to get which has the best prob hit/stand
def value_iteration(gamma=1.0, theta=0.001):
    states = [(player, dealer, usable_ace)
            for player in range(4, 22)
            for dealer in range(1, 11)
            for usable_ace in [True, False]]

    V = {s: 0 for s in states}
    memo_dealer = {}
    policy = {}
    while True:
        delta = 0
        for s in states:
            player, dealer, ace = s
            if player > 21:
                policy[s] = 0 
                continue
            v = V[s]
            hit = hit_value(V, player, dealer,gamma)
            stick, _ = stand_value(player, dealer, gamma, memo_dealer)
            V[s] = max(hit, stick)
            policy[s] = 1 if hit > stick else 0 
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    return policy

def dp_plot(policy, title, reverse):
    players = range(4, 22)
    dealers = range(1, 11)

    grid = np.zeros((len(players), len(dealers)))

    for i, p in enumerate(players):
        for j, d in enumerate(dealers):
            grid[i, j] = reverse * (policy[(p, d, False)])  # 0=stick, 1=hit

    plt.imshow(grid)
    plt.xticks(range(len(dealers)), dealers)
    plt.yticks(range(len(players)), players)
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    plt.title(title)
    plt.colorbar()
    plt.show()

