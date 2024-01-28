import torch
from queue import Queue

class RolloutPointClouds:

    def __init__(self, num_envs, num_transitions_per_env, pointclouds_shape, actions_shape, device='cpu'):

        self.device = device
        self.actions_shape = actions_shape
        self.pointclouds_shape = pointclouds_shape
        self.pointclouds = torch.zeros(num_envs, num_transitions_per_env * pointclouds_shape, 4+self.actions_shape, device=self.device)
        self.labels = torch.zeros(num_envs, 32, device=self.device)
        self.dones = torch.zeros(num_envs, num_transitions_per_env, device=self.device).byte()

        # For Adapter
        self.pc_queue = Queue(maxsize=num_transitions_per_env)
        self.action_queue = Queue(maxsize=num_transitions_per_env)
        self.dones_queue = Queue(maxsize=num_transitions_per_env)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.timestamps = torch.tensor([0, 1, 2, 3],device=self.device)

        self.step = 0

    def add_transitions(self, pointclouds, actions, dones, labels):
        queue_is_full = False
        if self.pc_queue.full() or self.dones_queue.full():
            # 如果队列满了，丢弃最老的元素
            queue_is_full = True
            self.pc_queue.get()
            self.dones_queue.get()
            self.action_queue.get()

        self.pc_queue.put(pointclouds)
        self.action_queue.put(actions)
        self.dones_queue.put(dones)
        self.labels = labels
        
        #self.step += 1
        return queue_is_full



    def mini_batch_generator(self, mini_batch_size):

        for i in range(self.num_transitions_per_env):

            self.dones[:,i] = self.dones_queue.queue[i]
            self.pointclouds[:,self.pointclouds_shape*i : self.pointclouds_shape * (i+1) ,0:3] = self.pc_queue.queue[i]
            self.pointclouds[:,self.pointclouds_shape*i : self.pointclouds_shape * (i+1) ,3] = self.timestamps[i]
            self.pointclouds[:,self.pointclouds_shape*i : self.pointclouds_shape * (i+1) ,4:11] = self.action_queue.queue[i].unsqueeze(1).repeat(1, self.pointclouds_shape, 1)

        pointcloud_batch = torch.zeros((mini_batch_size, 7), device=self.device)
        labels_batch = torch.zeros((mini_batch_size, 32), device=self.device)
        index = torch.nonzero(torch.all(self.dones==0,dim=1)).squeeze()
        #print(index)
        if len(index.size()) == 1 and index.size(0) > mini_batch_size - 1:
            random_indices = torch.randint(0, len(index), (mini_batch_size,)) # 从张量中获取随机抽取的两个元素
            pointcloud_batch = self.pointclouds[index[random_indices],:,:]
            labels_batch = self.labels[index[random_indices],:]
        else:
            return None, None
        
        return pointcloud_batch, labels_batch


if __name__ == '__main__':
    device = "cuda:0"
    num_envs = 3
    mini_batch_size = 2

    storage = RolloutPointClouds(num_envs, 4, 10, 7, device=device)

    feat1 = torch.ones((num_envs, 7), device=device)
    pcs = torch.ones(num_envs,10,3).uniform_(0,1).to(device)
    dones = torch.zeros(num_envs, device=device)
    dones1 = torch.ones(num_envs, device=device)
    labels = torch.ones((num_envs, 32), device=device)



    storage.add_transitions(pcs,feat1,dones,labels)
    storage.add_transitions(pcs,feat1,dones,labels)
    storage.add_transitions(pcs,feat1,dones,labels)
    storage.add_transitions(pcs,feat1,dones,labels)
    #print(storage.action_queue.queue)

    storage.add_transitions(pcs,feat1,dones,labels)
    #print(storage.action_queue.queue)

    pointcloud_batch, labels_batch = storage.mini_batch_generator(mini_batch_size)
    if pointcloud_batch is not None:
        print(pointcloud_batch.size())
        print(labels_batch.size())
    else:
        print("Not found")



