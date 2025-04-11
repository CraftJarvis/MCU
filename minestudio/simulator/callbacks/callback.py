
class MinecraftCallback:
    
    def before_step(self, sim, action):
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        return obs, reward, terminated, truncated, info
    
    def before_reset(self, sim, reset_flag: bool) -> bool: # whether need to call env reset
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        return obs, info
    
    def before_close(self, sim):
        return
    
    def after_close(self, sim):
        return
    
    def before_render(self, sim):
        return
    
    def after_render(self, sim):
        return
