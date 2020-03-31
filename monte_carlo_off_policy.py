import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from blackjack_rl import ActionSpace, BlackJackGame, BlackJackState

class StochasticPolicyAction:
    def __init__(self):
        self.num_of_actions = len(ActionSpace().action_space)
        self.actions_prob = np.zeros(self.num_of_actions)
        self.actions_prob_cumm = np.zeros(self.num_of_actions)
        increment = 1/float(self.num_of_actions)
        # init action probabilities to be equal for each action
        for i in range(self.num_of_actions):
            self.actions_prob[i] = increment
        self.update_cumm_prob()

    def update_cumm_prob(self):
        self.actions_prob_cumm[0] = self.actions_prob[0]
        for i in range(1,self.num_of_actions):
            self.actions_prob_cumm[i] = self.actions_prob[i]+self.actions_prob_cumm[i-1]
    
    def choose_action(self):
        action_chosen = np.random.rand()
        ind = 0
        while action_chosen > self.actions_prob_cumm[ind]:
            ind += 1
        return ActionSpace().action_space[ind]

class StochasticPolicy:
    def __init__(self):
        self.state_actions = [StochasticPolicyAction()]*200
    
    def state_to_index(self,state):
        # there are 200 states:
        # 2 options for usable ace or no usable ace
        # 10 options for player cards sum: 12 - 21
        # 10 options for dealer cards sum (at the beginning): 2 - 11
        ind = 0
        if state.usable_ace:
            ind = 100
        ind += (state.player_sum - 12)*10
        ind += state.dealer_sum - 2

        return ind

    def choose_action(self,state):
        return self.state_actions[self.state_to_index(state)].choose_action()

class DeterministicPolicy(StochasticPolicy):
    def __init__(self):
        super().__init__()
        state = BlackJackState()
        # stick for cards_sum 20 and 21, otherwise hit
        for usable_ace in [True,False]:
            for dealer_sum in range(2,12):
                for player_sum in range(12,20):
                    state.initialize_state(usable_ace,player_sum,dealer_sum)
                    sto_action = StochasticPolicyAction()
                    sto_action.actions_prob[ActionSpace().action_space.index(ActionSpace().hit)] = 1
                    sto_action.actions_prob[ActionSpace().action_space.index(ActionSpace().stick)] = 0
                    sto_action.update_cumm_prob()
                    self.state_actions[self.state_to_index(state)] = sto_action
                for player_sum in range(20,22):
                    state.initialize_state(usable_ace,player_sum,dealer_sum)
                    sto_action = StochasticPolicyAction()
                    sto_action.actions_prob[ActionSpace().action_space.index(ActionSpace().hit)] = 1
                    sto_action.actions_prob[ActionSpace().action_space.index(ActionSpace().stick)] = 0
                    sto_action.update_cumm_prob()
                    self.state_actions[self.state_to_index(state)] = sto_action
    
    def update_from_action_value(self,state_ind,action_values):
        for i in range(StochasticPolicyAction().num_of_actions):
            self.state_actions[state_ind].actions_prob[i] = 0
        ind = np.argmax(action_values)
        self.state_actions[state_ind].actions_prob[ind] = 1
        self.state_actions[state_ind].update_cumm_prob()

class MonteCarloOffPolicy:
    def __init__(self):
        self.target_policy = DeterministicPolicy()
        self.behavior_policy = StochasticPolicy()
        self.environment = BlackJackGame()
        self.q = np.zeros((len(StochasticPolicy().state_actions),StochasticPolicyAction().num_of_actions)) # action value approximation 
        self.c = np.zeros((len(StochasticPolicy().state_actions),StochasticPolicyAction().num_of_actions)) # cumulative sum of the weights
        self.discount = 1

    def find_determinstic_policy(self,num_of_episodes):
        per_jump = 0.01
        percentage = per_jump
        for episode in range(num_of_episodes):
            w = 1.0 # importance sampling ratio
            g = 0.0 # the return
            state_history, action_history, reward_history = self.run_episode()
            for step in range(len(action_history)-1,-1,-1):
                state_ind = StochasticPolicy().state_to_index(state_history[step])
                action_ind = ActionSpace().action_space.index(action_history[step])
                g = self.discount*g + reward_history[step]
                self.c[state_ind,action_ind] += w
                self.q[state_ind,action_ind] += (w/self.c[state_ind][action_ind])*(g-self.q[state_ind][action_ind])
                self.target_policy.update_from_action_value(state_ind,self.q[state_ind])
                if action_history[step] != self.target_policy.choose_action(state_history[step]):
                    break
                w *= 1/self.behavior_policy.state_actions[state_ind].actions_prob[action_ind]

                # print progress
                if episode/num_of_episodes >= percentage:
                    print("done running " + str(int(percentage*100)) + "% of episodes")
                    percentage += per_jump
        print("done running 100% of episodes")

    def run_episode(self):
        state_history = []
        action_history = []
        reward_history = []

        self.environment.run_new_episode()
        state = self.environment.observation()

        done = False
        while not done:
            state_history.append(deepcopy(state))
            # make an action
            action = self.behavior_policy.choose_action(state)
            state, reward, done = self.environment.step(action)
            # update history
            action_history.append(action)
            reward_history.append(reward)
            
        return state_history, action_history, reward_history

    def plot_state_function_and_policies(self):
        x_v = np.array([[i]*10 for i in range(12,22)])
        y_v = np.array([[i for i in range(2,12)]]*10)
        z_no_ace = np.zeros((10,10))
        z_usable_ace = np.zeros((10,10))

        x_pi_no_ace_hit = []
        x_pi_no_ace_stick = []
        x_pi_usable_ace_hit = []
        x_pi_usable_ace_stick = []
        y_pi_no_ace_hit = []
        y_pi_no_ace_stick = []
        y_pi_usable_ace_hit = []
        y_pi_usable_ace_stick = []

        for player_sum in range(12,22):
            for dealer_sum in range(2,12):
                state = BlackJackState()
                state.initialize_state(False,player_sum,dealer_sum)
                state_ind = StochasticPolicy().state_to_index(state)
                z_no_ace[player_sum-12,dealer_sum-2] = max(self.q[state_ind])
                if self.q[state_ind,ActionSpace().action_space.index(ActionSpace().hit)] == \
                   z_no_ace[player_sum-12,dealer_sum-2]:
                    x_pi_no_ace_hit.append(dealer_sum)
                    y_pi_no_ace_hit.append(player_sum)
                else:
                    x_pi_no_ace_stick.append(dealer_sum)
                    y_pi_no_ace_stick.append(player_sum) 

                state.initialize_state(True,player_sum,dealer_sum)
                state_ind = StochasticPolicy().state_to_index(state)
                z_usable_ace[player_sum-12,dealer_sum-2] = max(self.q[state_ind])
                if self.q[state_ind,ActionSpace().action_space.index(ActionSpace().hit)] == \
                   z_usable_ace[player_sum-12,dealer_sum-2]:
                    x_pi_usable_ace_hit.append(dealer_sum)
                    y_pi_usable_ace_hit.append(player_sum)
                else:
                    x_pi_usable_ace_stick.append(dealer_sum)
                    y_pi_usable_ace_stick.append(player_sum) 

        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.set_title("v no usable ace")
        ax.plot_wireframe(x_v,y_v,z_no_ace)
        ax.set_xlim(12,21)
        ax.set_ylim(2,11)
        ax.set_zlim(-1,1)
        ax.set_xlabel('Player sum')
        ax.set_ylabel('Dealer sum')
        ax.set_zlabel('State value')

        ax = fig.add_subplot(223, projection='3d')
        ax.set_title("v usable ace")
        ax.plot_wireframe(x_v,y_v,z_usable_ace)
        ax.set_xlim(12,21)
        ax.set_ylim(2,11)
        ax.set_zlim(-1,1)
        ax.set_xlabel('Player sum')
        ax.set_ylabel('Dealer sum')
        ax.set_zlabel('State value')

        ax = fig.add_subplot(222)
        ax.set_title("policy no usable ace")
        ax.scatter(x_pi_no_ace_stick,y_pi_no_ace_stick,c='r',label='Stick')
        ax.scatter(x_pi_no_ace_hit,y_pi_no_ace_hit,c='g',label='Hit')
        ax.set_xticks(range(2,12))
        ax.set_yticks(range(12,22))
        ax.set_xlabel('Dealer sum')
        ax.set_ylabel('Player sum')
        ax.legend(loc='upper right')
        ax.grid()
        
        ax = fig.add_subplot(224)
        ax.set_title("policy usable ace")
        ax.scatter(x_pi_usable_ace_stick,y_pi_usable_ace_stick,c='r',label='Stick')
        ax.scatter(x_pi_usable_ace_hit,y_pi_usable_ace_hit,c='g',label='Hit')
        ax.set_xticks(range(2,12))
        ax.set_yticks(range(12,22))
        ax.set_xlabel('Dealer sum')
        ax.set_ylabel('Player sum')
        ax.legend(loc='upper right')
        ax.grid()

        plt.show()

if __name__ == "__main__":
    mc_bj = MonteCarloOffPolicy()
    mc_bj.find_determinstic_policy(5000000)
    mc_bj.plot_state_function_and_policies()