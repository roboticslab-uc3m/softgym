import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.misc import quatFromAxisAngle
import copy
from copy import deepcopy


class ClothHangEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_drop_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized


    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_cloth_size(self):
        return np.random.randint(60, 100), np.random.randint(60, 100)

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            #self.create_hanger()
            self.action_tool.reset([0., -1., 0.])

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            config['target_pos'] = self._get_flat_pos()
            self._set_to_vertical(x_low=np.random.random() * 0.2 - 0.1, height_low=np.random.random() * 0.1 + 0.1)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            #self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))#DESCOMENTAR?

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))
        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
            # self.action_tool.visualize_picker_boundary()
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        pyflex.step()
        return

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        r = - curr_dist
        return r

    def _get_info(self):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        performance = -curr_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init)}

    #def create_hanger(self, hang_dis_x, hang_dis_z, height, border):
    def create_hanger(self):
        """
        hang consists of three thin boxes in Flex making an arc
        dis_x: the length of the hang
        dis_z: the width of the hang
        height: the height of the hang.
        border: the thickness of the hang wall.

        the halfEdge determines the center point of each wall.
        Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
        That's why left and right walls have exactly the same params, and so do front and back walls.   
        """
        '''
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # floor
        halfEdge = np.array([hang_dis_x / 2. + border, border / 2., hang_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([border / 2., (height) / 2., hang_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])
        
        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes
        '''
        # Create hang directly

        halfEdge = np.array([0.15, 0.8, 0.15])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)


    # WIP - Make this func define params in action_space.py/set_picker_pos()
    def init_hanger_state(self, x, y, hang_dis_x, hang_dis_z, height, border):
        '''
        set the initial state of the hang.
        '''
        dis_x, dis_z = hang_dis_x, hang_dis_z
        x_center, y_curr, y_last = x, y, 0.
        if self.action_mode in ['sawyer', 'franka']:
            y_curr = y_last = 0.56 # NOTE: robotics table
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of 3 boxes forming the hang
        states = np.zeros((3, 14))
        '''
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat
        '''
        # right wall
        states[2, :3] = np.array([x_center + (dis_x + border) / 2., (height) / 2. + y_curr, 0.])
        states[2, 3:6] = np.array([x_center + (dis_x + border) / 2., (height) / 2. + y_last, 0.])

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states

    #Sets Hang params and calls parent set_scene
    #def set_scene(self, config, states=None):
    def set_hang(self, states=None):


        # Hard coded (pls change this to be available from config) - Hang params
        self.hang_dis_x = 0.6
        self.hang_dis_z = 0.1
        self.height = 0.6
        self.border = 0.1
        self.x_center = 0

        # create hang - calls pyflex.add_box()
        self.create_hanger(self.hang_dis_x, self.hang_dis_z, self.height, self.border)

        # move glass to be at ground or at the table
        self.hang_states = self.init_hanger_state(self.x_center, 0, self.hang_dis_x, self.hang_dis_z, self.height, self.border)

        #pyflex.set_shape_states(self.hang_states)

        # record hang floor center x
        #self.hang_x = self.x_center

        

        