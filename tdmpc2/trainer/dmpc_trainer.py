from trainer.offline_trainer import OfflineTrainer
from dmpc_agent import DMPCAgent # Import the new agent

# Note: We might need Buffer modifications if sequence sampling changes drastically
# from common.buffer import Buffer 
# from common.logger import Logger

class DMPCTrainer(OfflineTrainer):
    """
    Trainer class for D-MPC offline training.
    Inherits data loading and evaluation logic from OfflineTrainer,
    but uses the DMPCAgent for planning and updates.
    """

    # Override __init__ to instantiate the correct agent
    def __init__(self, cfg, env, agent, buffer, logger):
        print("Initializing DMPCTrainer")
        
        # Explicitly create the DMPCAgent, overriding whatever might be passed
        dmpc_agent = DMPCAgent(cfg)

        # Call the parent constructor (OfflineTrainer's init)
        # Pass the newly created dmpc_agent
        # The env, buffer, logger are passed through from the main script
        super().__init__(cfg=cfg, env=env, agent=dmpc_agent, buffer=buffer, logger=logger)
        print("DMPCTrainer initialized with DMPCAgent.")

    # The train method from OfflineTrainer can likely be reused directly.
    # It calls self._load_dataset() (which we modified for memory)
    # and then loops calling self.agent.update(self.buffer).
    # Since self.agent is now DMPCAgent, its update method will be called,
    # performing the correct D-MPC training steps.
    # The eval method from OfflineTrainer also calls self.agent.act(), 
    # which in DMPCAgent uses the SSR planner.

    # def train(self):
    #     # ... (Can usually inherit directly from OfflineTrainer) ...
    #     pass

    # def eval(self):
    #     # ... (Can usually inherit directly from OfflineTrainer) ...
    #     pass 