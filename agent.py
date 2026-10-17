import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions) # Explore
        else:
            return np.argmax(self.q_table[state])   # Exploit

    def update(self, state, action, reward, next_state, terminated):
        # Q-Learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (self.gamma * self.q_table[next_state][best_next_action] * (not terminated))
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)