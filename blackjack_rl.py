import numpy as np

class Deck:
    def draw_cards(self):
        # for our needs there are 13 types of cards
        # 1 - Ace - value of 11 or 1
        # 2-10 - 2 up to 10 - value of the number
        # 11-13 jack quin king - value of 10
        
        # assuming each card always has the probability of 1/13
        return np.random.randint(1,14)

class ActionSpace:
    def __init__(self):
        self.action_space = ["hit","stick"]
        self.hit = "hit"
        self.stick = "stick"

    def sample(self):
        action = np.random.randint(len(self.action_space))
        return self.action_space[action]

class PlayerState:
    def __init__(self):
        self.initialize_state()

    def initialize_state(self,usable_ace=False,cards_sum=0):
        self.usable_ace = usable_ace
        self.cards_sum = cards_sum

class BlackJackState:
    def __init__(self):
        self.initialize_state()

    def initialize_state(self,usable_ace=False,player_sum=0,dealer_sum=0):
        self.usable_ace = usable_ace
        self.player_sum = player_sum
        self.dealer_sum = dealer_sum

class Player:
    def __init__(self):
        self.state = PlayerState()

    def hit_card(self,card):
        # set card value
        if card == 1:
            if self.state.cards_sum <= 10:
                card = 11
                self.state.usable_ace = True
        elif card > 10:
            card = 10
        
        self.state.cards_sum += card
        if self.state.cards_sum > 21:
            if self.state.usable_ace:
                self.state.usable_ace = False
                self.state.cards_sum -= 10

class Dealer(Player):
    def __init__(self,threshold):
        super().__init__()
        self.thresh_val = threshold
    
    def playing_action(self,player_sum):
        if self.state.cards_sum > player_sum:
            return ActionSpace().stick
        elif self.state.cards_sum >= self.thresh_val and self.state.cards_sum == player_sum:
            return ActionSpace().stick
        return ActionSpace().hit

class BlackJackGame:
    def __init__(self):
        # hit is True, stick is False
        self.deck = Deck()
        self.dealer = Dealer(17)
        self.player = Player()
        self.state = BlackJackState()
        self.episode_ended = True
        self.reward = 0
        self.debug = False

    def run_new_episode(self,init_state=BlackJackState(),debug=False):
        self.state.initialize_state(init_state.usable_ace,init_state.player_sum,init_state.dealer_sum)
        self.dealer.state.initialize_state(init_state.dealer_sum==11,init_state.dealer_sum)
        self.player.state.initialize_state(init_state.usable_ace,init_state.player_sum)
        self.episode_ended = False
        self.reward = 0
        self.debug = debug
        
        if not init_state.player_sum:
            self.dealer.hit_card(self.deck.draw_cards())

        while self.player.state.cards_sum < 12:
            self.player.hit_card(self.deck.draw_cards())

        self.update_state()
        self.print_state()

    def update_state(self):
        self.state.usable_ace = self.player.state.usable_ace
        self.state.player_sum = self.player.state.cards_sum
        self.state.dealer_sum = self.dealer.state.cards_sum

    def step(self,action):
        if not self.episode_ended:
            if action == ActionSpace().hit:
                self.player.hit_card(self.deck.draw_cards())
                self.update_state()
                if self.state.player_sum > 21:
                    self.episode_ended = True
                    self.reward = -1
            else:
                while self.dealer.playing_action(self.state.player_sum) == ActionSpace().hit:
                    self.dealer.hit_card(self.deck.draw_cards())
                self.update_state()

                if self.state.dealer_sum > self.state.player_sum:
                    if self.state.dealer_sum > 21:
                        self.reward = 1
                    else:
                        self.reward = -1
                    
                elif self.state.dealer_sum < self.state.player_sum:
                    self.reward = 1
                else:
                    self.reward = 0

                self.episode_ended = True

        self.print_state()
        return self.state, self.reward, self.episode_ended
    
    def observation(self):
        self.print_state()
        return self.state

    def print_state(self):
        if self.debug:
            print("Current state:")
            if self.state.usable_ace:
                print("player has a usable ace")
            else:
                print("player has no usable ace")
            print("player cards sum: " + str(self.state.player_sum))
            print("dealer cards sum: " + str(self.state.dealer_sum))
            print("")