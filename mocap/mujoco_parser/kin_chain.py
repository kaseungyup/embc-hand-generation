import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_edges
from util import r2rpy,rpy2r,rodrigues,pr2t

class KinematicChainClass(object):
    def __init__(self,name='Kinematic Chain'):
        """
            Initialize Kinematic Chain Object
        """
        self.name        = name 
        self.chain       = None
        self.joint_names = []
        # Initialize chain
        self.init_chain()
        
    def init_chain(self):
        """
            Initialize chain
        """
        if self.chain is not None:
            self.chain.clear()
        self.chain = nx.DiGraph(name=self.name)
        
    def get_n_joint(self):
        """
            Get the number of joints
        """
        return self.chain.number_of_nodes()
    
    def get_joint_idx(self,joint_name):
        """
            Get the index of a joint
        """
        joint_idx = self.joint_names.index(joint_name)
        return joint_idx
    
    def get_joint_idxs(self,joint_names):
        """
            Get the indices of joints 
        """
        joint_idxs = [[idx for idx,item in enumerate(self.joint_names) 
                       if item==joint_name] for joint_name in joint_names]
        return joint_idxs
    
    def set_joint_q(self,joint_names,qs):
        """ 
            Set joint values
        """
        for (joint_name,q) in zip(joint_names,qs):
            self.chain.nodes[self.get_joint_idx(joint_name)]['q'] = q
            
    def set_joint_p(self,joint_name,p):
        """ 
            Set joint p
        """
        self.chain.nodes[self.get_joint_idx(joint_name)]['p'] = p
        
    def set_joint_R(self,joint_name,R):
        """ 
            Set joint R
        """
        self.chain.nodes[self.get_joint_idx(joint_name)]['R'] = R
        
    def set_joint_pR(self,joint_name,p,R):
        """ 
            Set joint p and R
        """
        self.set_joint_p(joint_name=joint_name,p=p)
        self.set_joint_R(joint_name=joint_name,R=R)
        
    def set_root_joint_p(self,p):
        """ 
            Set root joint p
        """
        joint_name = self.get_root_name()
        self.set_joint_p(joint_name=joint_name,p=p)
        
    def set_root_joint_R(self,R):
        """ 
            Set root joint R
        """
        joint_name = self.get_root_name()
        self.set_joint_R(joint_name=joint_name,R=R)
        
    def set_root_joint_pR(self,p,R):
        """ 
            Set root joint p and R
        """
        joint_name = self.get_root_name()
        self.set_joint_p(joint_name=joint_name,p=p)
        self.set_joint_R(joint_name=joint_name,R=R)
        
    def set_joint_p_offset(self,joint_name,p_offset):
        """ 
            Set joint p offset
        """
        self.chain.nodes[self.get_joint_idx(joint_name)]['p_offset'] = p_offset
        
    def set_joint_R_offset(self,joint_name,R_offset):
        """ 
            Set joint R offset
        """
        self.chain.nodes[self.get_joint_idx(joint_name)]['R_offset'] = R_offset
        
    def set_joint_R_offset_list(self,joint_name_list,R_offset_list):
        """ 
            Set joint R offset (list)
        """
        for (joint_name,R_offset) in zip(joint_name_list,R_offset_list):    
            self.set_joint_R_offset(joint_name=joint_name,R_offset=R_offset)

    def add_joint(self,name='',a=np.array([0,0,0]),
                  p=np.zeros(3),R=np.eye(3),
                  p_offset=np.zeros(3),R_offset=np.eye(3),
                  parent_name=None):
        """
            Add joint to the chain
        """
        # Add new node (=joint)
        new_joint_idx = self.get_n_joint()
        self.chain.add_node(new_joint_idx)
        
        # Update joint information
        joint_info = {'name':name,'p':p,'R':R,'q':0.0,
                      'a':a,'p_offset':p_offset,'R_offset':R_offset,
                      'parent':[],'childs':[]}
        self.chain.update(nodes=[(new_joint_idx,joint_info)])
        
        # Append joint name
        self.joint_names.append(name)
        
        # Add parent 
        if parent_name is not None:
            # Add parent index
            parent_idx = self.get_joint_idx(parent_name)
            self.chain.nodes[new_joint_idx]['parent'] = parent_idx
            # Connect parent and child
            self.chain.add_edge(u_of_edge=parent_idx,v_of_edge=new_joint_idx)
        
        # Append childs to the parent
        parent_idx = self.get_joint_idx(name)
        parent_childs = self.chain.nodes[parent_idx]['childs']
        parent_childs.append(new_joint_idx)
        
    def get_joint(self,joint_idx):
        """
            Get joint in tree
        """
        joint = self.chain.nodes[joint_idx]
        return joint
        
    def get_joint_p(self,joint_idx=None,joint_name=None):
        """
            Get joint p
        """
        if joint_idx is not None:
            return self.chain.nodes[joint_idx]['p']
        if joint_name is not None:
            return self.chain.nodes[self.get_joint_idx(joint_name)]['p']
        
    def get_joint_R(self,joint_idx=None,joint_name=None):
        """
            Get joint R
        """
        if joint_idx is not None:
            return self.chain.nodes[joint_idx]['R']
        if joint_name is not None:
            return self.chain.nodes[self.get_joint_idx(joint_name)]['R']
        
    def get_joint_pR(self,joint_idx=None,joint_name=None):
        """
            Get joint p
        """
        p = self.get_joint_p(joint_idx=joint_idx,joint_name=joint_name)
        R = self.get_joint_R(joint_idx=joint_idx,joint_name=joint_name)
        return p,R
    
    def get_root_joint_pR(self):
        """
            Get joint p
        """
        joint_name = self.get_root_name()
        p = self.get_joint_p(joint_idx=None,joint_name=joint_name)
        R = self.get_joint_R(joint_idx=None,joint_name=joint_name)
        return p,R
    
    def get_root_idx(self):
        """ 
            Get root index
        """
        root_idx = 0
        for node_idx,node in enumerate(self.chain.nodes):
            if self.chain.in_degree(node) == 0:
                root_idx = node_idx
                break
        return root_idx
    
    def get_root_name(self):
        """ 
            Get root name
        """
        return self.joint_names[self.get_root_idx()]
    
    def update_joint_info(self,joint_idx,key,value):
        """
            Update joint information 
        """
        self.chain.nodes[joint_idx][key] = value
        
    def forward_kinematics(self,root_idx=0):
        """
            Forward Kinematics
        """
        for idx,edge in enumerate(dfs_edges(self.chain,source=root_idx)):
            idx_fr   = edge[0]
            idx_to   = edge[1]
            joint_fr = self.get_joint(idx_fr)
            joint_to = self.get_joint(idx_to)
            # Update p
            p = joint_fr['R']@joint_to['p_offset'] + joint_fr['p']
            self.update_joint_info(idx_to,'p',p)
            # Update R
            a_to = joint_to['a']
            if abs(np.linalg.norm(a_to)-1) < 1e-6: # with axis
                q_to = joint_to['q']
                R = joint_fr['R']@joint_to['R_offset']@rodrigues(a=a_to,q_rad=q_to)
            else:
                R = joint_fr['R']@joint_to['R_offset']
            self.update_joint_info(idx_to,'R',R)
    
    
    def print_chain_info(self):
        """
            Print chain information
        """
        n_joint = self.get_n_joint()
        for j_idx in range(n_joint):
            joint = self.get_joint(joint_idx=j_idx)
            print ("[%d/%d] joint name:[%s] p:%s rpy_deg:%s"%
                   (j_idx,n_joint,joint['name'],
                    joint['p'],
                    np.degrees(r2rpy(joint['R']))
                   ))
        
    def plot_chain_graph(self,align='horizontal',figsize=(6,4),
                   node_size=300,font_size_node=10,node_colors=None,
                   font_size_title=10,ROOT_ON_TOP=True):
        """
            Plot chain graph
        """
        n_joint = self.get_n_joint()
        tree = self.chain
        for layer, nodes in enumerate(nx.topological_generations(tree)):
            for node in nodes:
                tree.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(tree,align=align,
                                     scale=1.0,subset_key='layer')
        # Invert the tree so that the root node comes on the top
        if ROOT_ON_TOP:
            pos = {node: (x, -y) for node, (x, y) in pos.items()} 
        # Plot
        fig,ax = plt.subplots(figsize=figsize)
        if node_colors is None:
            node_colors = []
            for j_idx in range(n_joint):
                a = self.get_joint(j_idx)['a']
                if np.linalg.norm(a) < 1e-6:
                    node_color = (1,1,1,0.5)
                else:
                    node_color = [0,0,0,0.5]
                    node_color[np.argmax(a)] = 1
                    node_color = tuple(node_color)
                node_colors.append(node_color)
        nx.draw_networkx(tree,pos=pos,ax=ax,with_labels=True,
                         node_size=node_size,font_size=font_size_node,
                         node_color=node_colors,
                         linewidths=1,edgecolors='k')
        ax.set_title('%s'%(tree.name),fontsize=font_size_title)
        fig.tight_layout()
        plt.show()
        
    def plot_chain_mujoco(
        self,env,
        PLOT_LINK=True,r_link=0.005,rgba_link=(0,0,0,0.5),
        PLOT_JOINT=True,PLOT_JOINT_AXIS=True,PLOT_JOINT_SPHERE=True,PLOT_JOINT_NAME=True,
        axis_len_joint=0.05,axis_width_joint=0.005,r_joint=0.01,rgba_joint=(0.1,0.1,0.1,0.9),
        PLOT_REV_AXIS=True,r_axis = 0.015,h_axis=0.1
        ):
        """ 
            Plot chain in MuJoCo
        """
        
        # Plot link
        if PLOT_LINK:
            for idx,edge in enumerate(dfs_edges(self.chain,source=0)):
                joint_fr = self.get_joint(edge[0])
                joint_to = self.get_joint(edge[1])
                env.plot_cylinder_fr2to(p_fr=joint_fr['p'],p_to=joint_to['p'],
                                        r=r_link,rgba=rgba_link)
            
        # Plot joint
        if PLOT_JOINT:
            for j_idx in range(self.get_n_joint()):
                joint = self.get_joint(j_idx)
                if PLOT_JOINT_NAME:
                    joint_name = joint['name']
                else:
                    joint_name = ''
                env.plot_T(p=joint['p'],R=joint['R'],
                        PLOT_AXIS=PLOT_JOINT_AXIS,axis_len=axis_len_joint,axis_width=axis_width_joint,
                        PLOT_SPHERE=PLOT_JOINT_SPHERE,sphere_r=r_joint,sphere_rgba=rgba_joint,
                        label=joint_name)
        
        # Plot revolute axis
        if PLOT_REV_AXIS:
            for j_idx in range(self.get_n_joint()):
                joint = self.get_joint(j_idx)
                a = joint['a']
                if np.linalg.norm(a) > 1e-6:
                    p,R = joint['p'],joint['R']
                    p2 = p + R@a*h_axis
                    axis_color = [0,0,0,0.5]
                    axis_color[np.argmax(a)] = 1
                    env.plot_arrow_fr2to(p_fr=p,p_to=p2,r=r_axis,rgba=axis_color)
                    
def get_T_joi_from_chain_nc(chain):
    """ 
        Get joints of interest of NC mocap chain
    """
    
    p_rs,R_rs = chain.get_joint_pR(joint_name='upperarm_r')
    p_re,R_re = chain.get_joint_pR(joint_name='lowerarm_r')
    p_rw,R_rw = chain.get_joint_pR(joint_name='hand_r')
    
    p_ls,R_ls = chain.get_joint_pR(joint_name='upperarm_l')
    p_le,R_le = chain.get_joint_pR(joint_name='lowerarm_l')
    p_lw,R_lw = chain.get_joint_pR(joint_name='hand_l')
    
    p_neck = 0.5*(p_rs+p_ls) # neck position to be the center of two shoulder positions
    
    p_rp,R_rp = chain.get_joint_pR(joint_name='thigh_r')
    p_rk,R_rk = chain.get_joint_pR(joint_name='calf_r')
    p_ra,R_ra = chain.get_joint_pR(joint_name='foot_r')
    
    p_lp,R_lp = chain.get_joint_pR(joint_name='thigh_l')
    p_lk,R_lk = chain.get_joint_pR(joint_name='calf_l')
    p_la,R_la = chain.get_joint_pR(joint_name='foot_l')
    
    # Right hand
    p_rthumb_1 = chain.get_joint_p(joint_name='thumb_01_r')
    p_rthumb_2 = chain.get_joint_p(joint_name='thumb_02_r')
    p_rthumb_3 = chain.get_joint_p(joint_name='thumb_03_r')
    
    p_rindex_0 = chain.get_joint_p(joint_name='index_metacarpal_r')
    p_rindex_1 = chain.get_joint_p(joint_name='index_01_r')
    p_rindex_2 = chain.get_joint_p(joint_name='index_02_r')
    p_rindex_3 = chain.get_joint_p(joint_name='index_03_r')
    
    p_rmiddle_0 = chain.get_joint_p(joint_name='middle_metacarpal_r')
    p_rmiddle_1 = chain.get_joint_p(joint_name='middle_01_r')
    p_rmiddle_2 = chain.get_joint_p(joint_name='middle_02_r')
    p_rmiddle_3 = chain.get_joint_p(joint_name='middle_03_r')
    
    p_rring_0 = chain.get_joint_p(joint_name='ring_metacarpal_r')
    p_rring_1 = chain.get_joint_p(joint_name='ring_01_r')
    p_rring_2 = chain.get_joint_p(joint_name='ring_02_r')
    p_rring_3 = chain.get_joint_p(joint_name='ring_03_r')
    
    p_rpinky_0 = chain.get_joint_p(joint_name='pinky_metacarpal_r')
    p_rpinky_1 = chain.get_joint_p(joint_name='pinky_01_r')
    p_rpinky_2 = chain.get_joint_p(joint_name='pinky_02_r')
    p_rpinky_3 = chain.get_joint_p(joint_name='pinky_03_r')
    
    # Left hand
    p_lthumb_1 = chain.get_joint_p(joint_name='thumb_01_l')
    p_lthumb_2 = chain.get_joint_p(joint_name='thumb_02_l')
    p_lthumb_3 = chain.get_joint_p(joint_name='thumb_03_l')
    
    p_lindex_0 = chain.get_joint_p(joint_name='index_metacarpal_l')
    p_lindex_1 = chain.get_joint_p(joint_name='index_01_l')
    p_lindex_2 = chain.get_joint_p(joint_name='index_02_l')
    p_lindex_3 = chain.get_joint_p(joint_name='index_03_l')
    
    p_lmiddle_0 = chain.get_joint_p(joint_name='middle_metacarpal_l')
    p_lmiddle_1 = chain.get_joint_p(joint_name='middle_01_l')
    p_lmiddle_2 = chain.get_joint_p(joint_name='middle_02_l')
    p_lmiddle_3 = chain.get_joint_p(joint_name='middle_03_l')
    
    p_lring_0 = chain.get_joint_p(joint_name='ring_metacarpal_l')
    p_lring_1 = chain.get_joint_p(joint_name='ring_01_l')
    p_lring_2 = chain.get_joint_p(joint_name='ring_02_l')
    p_lring_3 = chain.get_joint_p(joint_name='ring_03_l')
    
    p_lpinky_0 = chain.get_joint_p(joint_name='pinky_metacarpal_l')
    p_lpinky_1 = chain.get_joint_p(joint_name='pinky_01_l')
    p_lpinky_2 = chain.get_joint_p(joint_name='pinky_02_l')
    p_lpinky_3 = chain.get_joint_p(joint_name='pinky_03_l')
    
    T_joi = {
        'hip': pr2t(chain.get_joint_p(joint_name='pelvis'),chain.get_joint_R(joint_name='pelvis')),
        'spine': pr2t(chain.get_joint_p(joint_name='spine_02'),chain.get_joint_R(joint_name='spine_02')),
        'neck': pr2t(p_neck,np.eye(3,3)),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'rthumb_1': pr2t(p_rthumb_1,np.eye(3,3)),
        'rthumb_2': pr2t(p_rthumb_2,np.eye(3,3)),
        'rthumb_3': pr2t(p_rthumb_3,np.eye(3,3)),
        'rindex_0': pr2t(p_rindex_0,np.eye(3,3)),
        'rindex_1': pr2t(p_rindex_1,np.eye(3,3)),
        'rindex_2': pr2t(p_rindex_2,np.eye(3,3)),
        'rindex_3': pr2t(p_rindex_3,np.eye(3,3)),
        'rmiddle_0': pr2t(p_rmiddle_0,np.eye(3,3)),
        'rmiddle_1': pr2t(p_rmiddle_1,np.eye(3,3)),
        'rmiddle_2': pr2t(p_rmiddle_2,np.eye(3,3)),
        'rmiddle_3': pr2t(p_rmiddle_3,np.eye(3,3)),
        'rring_0': pr2t(p_rring_0,np.eye(3,3)),
        'rring_1': pr2t(p_rring_1,np.eye(3,3)),
        'rring_2': pr2t(p_rring_2,np.eye(3,3)),
        'rring_3': pr2t(p_rring_3,np.eye(3,3)),
        'rpinky_0': pr2t(p_rpinky_0,np.eye(3,3)),
        'rpinky_1': pr2t(p_rpinky_1,np.eye(3,3)),
        'rpinky_2': pr2t(p_rpinky_2,np.eye(3,3)),
        'rpinky_3': pr2t(p_rpinky_3,np.eye(3,3)),
        'lthumb_1': pr2t(p_lthumb_1,np.eye(3,3)),
        'lthumb_2': pr2t(p_lthumb_2,np.eye(3,3)),
        'lthumb_3': pr2t(p_lthumb_3,np.eye(3,3)),
        'lindex_0': pr2t(p_lindex_0,np.eye(3,3)),
        'lindex_1': pr2t(p_lindex_1,np.eye(3,3)),
        'lindex_2': pr2t(p_lindex_2,np.eye(3,3)),
        'lindex_3': pr2t(p_lindex_3,np.eye(3,3)),
        'lmiddle_0': pr2t(p_lmiddle_0,np.eye(3,3)),
        'lmiddle_1': pr2t(p_lmiddle_1,np.eye(3,3)),
        'lmiddle_2': pr2t(p_lmiddle_2,np.eye(3,3)),
        'lmiddle_3': pr2t(p_lmiddle_3,np.eye(3,3)),
        'lring_0': pr2t(p_lring_0,np.eye(3,3)),
        'lring_1': pr2t(p_lring_1,np.eye(3,3)),
        'lring_2': pr2t(p_lring_2,np.eye(3,3)),
        'lring_3': pr2t(p_lring_3,np.eye(3,3)),
        'lpinky_0': pr2t(p_lpinky_0,np.eye(3,3)),
        'lpinky_1': pr2t(p_lpinky_1,np.eye(3,3)),
        'lpinky_2': pr2t(p_lpinky_2,np.eye(3,3)),
        'lpinky_3': pr2t(p_lpinky_3,np.eye(3,3)),
    }
    return T_joi
    