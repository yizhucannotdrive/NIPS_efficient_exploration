
import numpy as np
import gym

def main():
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    train_data = []
    num_train_data = 100000
    while len(train_data) < num_train_data:
        state = env.reset()
        step = 0
        while True:
            step += 1
            action =np.random.randint(2)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    main()