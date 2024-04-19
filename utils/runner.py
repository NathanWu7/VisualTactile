class Runner:
    def __init__(self, task, train_cfg) -> None:
        self.task = task
        self.train_cfg = train_cfg
        self.iterations = train_cfg["learn"]["max_iterations"] 
        
    def run(self):
        self.task.run(num_learning_iterations=self.iterations, log_interval=self.train_cfg["learn"]["save_interval"])

    def eval(self):
        self.task.eval(1000)

        
    