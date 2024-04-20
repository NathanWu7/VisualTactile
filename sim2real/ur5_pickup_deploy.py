from RL.test.test import TEST



if __name__ == "__main__":
    device = "cuda:0"
    policy = TEST(env, device)
    max_ep_length = 500
    
    for i in range(max_ep_length):
        policy.run()
