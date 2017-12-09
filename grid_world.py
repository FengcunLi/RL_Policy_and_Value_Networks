import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
class Block(object):
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x, self.y = coordinates[0], coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class GridWorld():
    def __init__(self, size):
        self.x_size = size
        self.y_size = size
        self.action_num = 4
        self.block_num = 7
        self.blocks = None
        self.available_grids = None
        self.state = None

    def get_positions(self, num):
        indices = np.random.choice(np.arange(len(self.available_grids)), size=num, replace=False)
        grids = [self.available_grids[index] for index in indices]
        for grid in grids:
            self.available_grids.remove(grid)
        return grids

    def reset(self):
        self.available_grids = [(x, y) for x in range(self.x_size) for y in range(self.y_size)]
        postions = self.get_positions(self.block_num)
        size = [1] * self.block_num
        intensity = [1] * self.block_num
        channel = [2, 1, 0, 1, 0, 1, 1]
        reward = [None, 1, -1, 1, -1, 1, 1]
        name = ["hero", "goal", "fire", "goal", "fire", "goal", "goal"]
        self.blocks = [Block(*args) for args in zip(postions, size, intensity, channel, reward, name)]
        self.state = self.render()
        return self.state
    
    def move(self, direction):
        hero = self.blocks[0]
        if direction == 0 and hero.y >=1:
            hero.y -= 1
        elif direction == 1 and hero.y <= self.y_size - 2:
            hero.y += 1
        elif direction == 2 and hero.x >= 1:
            hero.x -= 1
        elif direction == 3 and hero.x <= self.x_size - 2:
            hero.x += 1
        else:
            pass

    def check_hit(self):
        hero = self.blocks.pop(0)
        for block in self.blocks:
            if (hero.x == block.x) and (hero.y == block.y):
                self.available_grids.append([block.x, block.y])
                self.blocks.remove(block)
                if block.name == "goal":
                    self.blocks.append(Block(*self.get_positions(1), 1, 1, 1, 1, "goal"))
                elif block.name == "fire":
                    self.blocks.append(Block(*self.get_positions(1), 1, 1, 0, -1, "fire"))
                else:
                    raise
                self.blocks.insert(0, hero)
                return block.reward, False
        
        self.blocks.insert(0, hero)
        return 0.0, False

    def render(self):
        canvas = np.ones([self.y_size+2, self.x_size+2, 3])
        canvas[1:-1,1:-1,:] = 0
        for block in self.blocks:
            canvas[block.y+1:block.y+block.size+1,block.x+1:block.x+block.size+1,block.channel] = block.intensity
        r = scipy.misc.imresize(canvas[:,:,0], [84,84,1], interp='nearest')
        g = scipy.misc.imresize(canvas[:,:,1], [84,84,1], interp='nearest')
        b = scipy.misc.imresize(canvas[:,:,2], [84,84,1], interp='nearest')
        return np.stack([r, g, b],axis=2)

    def step(self, action):
        self.move(action)
        reward, done = self.check_hit()
        self.state = self.render()
        return self.state, reward, done
    
    def plot(self):
        plt.imshow(self.state, interpolation="nearest")
if __name__ == "__main__":
    env = GridWorld(size=5)
    _ = env.reset()
    env.plot()