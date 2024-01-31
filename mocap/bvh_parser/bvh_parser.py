import sys
sys.path.append("../mujoco_parser")
from util import rpy2r,t2p,t2r,pr2t,t2pr,rpy2r_order
from kin_chain import KinematicChainClass

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from skeleton import process_bvhfile,process_bvhkeyframe

def get_skeleton_from_bvh(bvh_path):
    """ 
        Get skeleton from bvh file
    """
    # Load
    skeleton = process_bvhfile(bvh_path)
    # FK
    for tick in range(skeleton.frames):
        new_frame = process_bvhkeyframe(
            skeleton.keyframes[tick],
            skeleton.root,
            skeleton.dt*tick,
        )
    return skeleton

def get_chains_from_skeleton(
    skeleton,
    env=None,
    rpy_order=[2,1,0],
    p_rate=0.1, # positional scale
    PLOT_CHAIN_GRAPH=True,
    PLOT_INIT_CHAIN=True,
    VERBOSE=True,
    ):
    """ 
        Get chains from skeleton
    """
    L = skeleton.frames
    skel_pos_info,skel_pos_time_list = skeleton.get_frames_worldpos()
    skel_rot_info,skel_rot_time_list = skeleton.get_frames_rotations()
    secs = np.array(skel_pos_time_list)[:,0] # [L]
    skel_pos_array     = np.array(skel_pos_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]
    skel_rpy_deg_array = np.array(skel_rot_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]

    # Initialize kinematic chain
    chain = KinematicChainClass(name='CMU-mocap')
    node = skeleton.root # root node
    chain.add_joint(
        name=node.name,
        p=np.array([0,0,0]),
        R=rpy2r(np.radians([0,0,0])),
    )
    deq = deque()
    while (len(node.children)>0) or (len(deq)>0):
        if len(deq) > 0:
            node = deq.pop()
            chain.add_joint(
                name = node.name, 
                parent_name = node.parent.name,
                p_offset = t2p(np.array(node.stransmat))*p_rate,
                R_offset = t2r(np.array(node.stransmat)),
            )
        for child in node.children:
            deq.append(child)
    
    # Copy init chain
    chain_init = deepcopy(chain)
    
    # Root joint name
    root_name = chain.get_root_name()
    
    # Plot chain graph
    if PLOT_CHAIN_GRAPH:
        chain.plot_chain_graph(
            align='vertical',
            figsize=(5,2),
            node_size=200,
            font_size_node=8,
            node_colors=None,
            font_size_title=10,
            ROOT_ON_TOP=True,
        )
        
    # Print joint names
    if VERBOSE:
        print ("[Joint names]")
        for j_idx,joint_name in enumerate(chain.joint_names):
            print ("[%02d/%d] joint_name:[%s]"%(j_idx,chain.get_n_joint(),joint_name))
        
    if PLOT_INIT_CHAIN:
        env.init_viewer(
            viewer_title="Kinematic Chain",
            viewer_width=1200,
            viewer_height=800,
            viewer_hide_menus=True,
        )
        env.update_viewer(
            azimuth=177,distance=4.0,elevation=-28,lookat=[0.0,0.1,0.65],
            VIS_TRANSPARENT=False,VIS_CONTACTPOINT=True,
            contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
            VIS_JOINT=True,jointlength=0.5,jointwidth=0.1,
            jointrgba=[0.2,0.6,0.8,0.6],
        )
        env.reset()
        chain.set_root_joint_pR(p=np.array([0,0,0]),R=rpy2r(np.radians([0,0,0])))
        chain.forward_kinematics() # forward kinematics chain
        while env.is_viewer_alive():
            chain.plot_chain_mujoco(
                env,
                r_link=0.04,
                rgba_link=(0.45, 0.25, 0.8, 0.5),
                PLOT_JOINT=True,PLOT_JOINT_AXIS=True,PLOT_JOINT_SPHERE=False,PLOT_JOINT_NAME=False,
                axis_len_joint=0.075,
                axis_width_joint=0.01,
                PLOT_REV_AXIS=False
            )
            env.plot_T(p=np.zeros(3),R=np.eye(3,3),PLOT_AXIS=True,axis_len=1.0,axis_width=0.005)
            env.render()
            # Save image
            if (env.render_tick%10)==0: scene_img = env.grab_image()
            # Terminate
            if env.render_tick >= 10: break; 
        # Close viewer
        env.close_viewer()
        # Plot
        plt.figure(figsize=(5,4));plt.imshow(scene_img)
        plt.title("Initial Chain",fontsize=10);plt.axis('off');plt.show()
        print ("Done.")
        
    # Make chains
    chains = []
    for tick in range(L):
        p_root = p_rate*skel_pos_array[tick,0,:]
        R_offset_list = []
        for joint_idx in range(chain.get_n_joint()):
            if joint_idx == 0: # root 
                rpy_deg = np.array([0,0,0])
                rpy_deg_root = skel_rpy_deg_array[tick,joint_idx,:] # backup root rot
            else:
                rpy_deg = skel_rpy_deg_array[tick,joint_idx,:]
            R_offset_list.append(
                rpy2r_order(np.radians(rpy_deg),order=rpy_order)
            )
            
        R_root = rpy2r_order(np.radians(rpy_deg_root),order=rpy_order)
        chain.set_root_joint_R(R=R_root)
        chain.set_root_joint_p(p=p_root)
        
        chain.set_joint_R_offset_list(chain.joint_names,R_offset_list)
        chain.forward_kinematics()
        # Append
        chains.append(deepcopy(chain)) # we have to 'deepcopy'
    
    # Return
    return secs,chains

def get_chains_zup(
    chains,
    T_trans_zup = pr2t(np.array([0,0,0]),rpy2r(np.radians([90,0,0]))),
    ):
    """ 
        Get chains from y-up to z-up
    """
    chains_zup = []
    for tick,chain in enumerate(chains):
        p_root,R_root = chain.get_root_joint_pR()
        T_root = pr2t(p_root,R_root)
        T_root_zup = T_trans_zup @ T_root
        p_root_zup,R_root_zup = t2pr(T_root_zup)
        # FK chain
        chain_zup = deepcopy(chain)
        chain_zup.set_root_joint_pR(p=p_root_zup,R=R_root_zup)
        chain_zup.forward_kinematics()
        chains_zup.append(chain_zup) # we have to 'deepcopy'
    # Return        
    return chains_zup


def get_chains_from_skeleton_with_predicted_hands(
    skeleton,
    env=None,
    predicted_hands=None,
    rpy_order=[2,1,0],
    p_rate=0.1, # positional scale
    PLOT_CHAIN_GRAPH=True,
    PLOT_INIT_CHAIN=True,
    VERBOSE=True,
    ):
    """ 
        Get chains from skeleton
    """
    L = skeleton.frames
    skel_pos_info,skel_pos_time_list = skeleton.get_frames_worldpos()
    skel_rot_info,skel_rot_time_list = skeleton.get_frames_rotations()
    secs = np.array(skel_pos_time_list)[:,0] # [L]
    skel_pos_array     = np.array(skel_pos_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]
    skel_rpy_deg_array = np.array(skel_rot_time_list)[:,1:].reshape(L,-1,3) # [L x n_joint x 3]
    hand_tip_idx = [24,29,34,39,43,52,57,62,67,71]
    hand_tip_pos = skel_pos_array[:,hand_tip_idx,:]
    hand_idx = [20,21,22,25,26,27,30,31,32,35,36,37,40,41,48,49,50,53,54,55,58,59,60,63,64,65,68,69]
    if predicted_hands is not None: skel_rpy_deg_array[:,hand_idx,:] = predicted_hands

    # Initialize kinematic chain
    chain = KinematicChainClass(name='CMU-mocap')
    node = skeleton.root # root node
    chain.add_joint(
        name=node.name,
        p=np.array([0,0,0]),
        R=rpy2r(np.radians([0,0,0])),
    )
    deq = deque()
    while (len(node.children)>0) or (len(deq)>0):
        if len(deq) > 0:
            node = deq.pop()
            chain.add_joint(
                name = node.name, 
                parent_name = node.parent.name,
                p_offset = t2p(np.array(node.stransmat))*p_rate,
                R_offset = t2r(np.array(node.stransmat)),
            )
        for child in node.children:
            deq.append(child)
    
    # Copy init chain
    chain_init = deepcopy(chain)
    
    # Root joint name
    root_name = chain.get_root_name()
    
    # Plot chain graph
    if PLOT_CHAIN_GRAPH:
        chain.plot_chain_graph(
            align='vertical',
            figsize=(5,2),
            node_size=200,
            font_size_node=8,
            node_colors=None,
            font_size_title=10,
            ROOT_ON_TOP=True,
        )
        
    # Print joint names
    if VERBOSE:
        print ("[Joint names]")
        for j_idx,joint_name in enumerate(chain.joint_names):
            print ("[%02d/%d] joint_name:[%s]"%(j_idx,chain.get_n_joint(),joint_name))
        
    if PLOT_INIT_CHAIN:
        env.init_viewer(
            viewer_title="Kinematic Chain",
            viewer_width=1200,
            viewer_height=800,
            viewer_hide_menus=True,
        )
        env.update_viewer(
            azimuth=177,distance=4.0,elevation=-28,lookat=[0.0,0.1,0.65],
            VIS_TRANSPARENT=False,VIS_CONTACTPOINT=True,
            contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
            VIS_JOINT=True,jointlength=0.5,jointwidth=0.1,
            jointrgba=[0.2,0.6,0.8,0.6],
        )
        env.reset()
        chain.set_root_joint_pR(p=np.array([0,0,0]),R=rpy2r(np.radians([0,0,0])))
        chain.forward_kinematics() # forward kinematics chain
        while env.is_viewer_alive():
            chain.plot_chain_mujoco(
                env,
                r_link=0.04,
                rgba_link=(0.45, 0.25, 0.8, 0.5),
                PLOT_JOINT=True,PLOT_JOINT_AXIS=True,PLOT_JOINT_SPHERE=False,PLOT_JOINT_NAME=False,
                axis_len_joint=0.075,
                axis_width_joint=0.01,
                PLOT_REV_AXIS=False
            )
            env.plot_T(p=np.zeros(3),R=np.eye(3,3),PLOT_AXIS=True,axis_len=1.0,axis_width=0.005)
            env.render()
            # Save image
            if (env.render_tick%10)==0: scene_img = env.grab_image()
            # Terminate
            if env.render_tick >= 10: break; 
        # Close viewer
        env.close_viewer()
        # Plot
        plt.figure(figsize=(5,4));plt.imshow(scene_img)
        plt.title("Initial Chain",fontsize=10);plt.axis('off');plt.show()
        print ("Done.")
        
    # Make chains
    chains = []
    for tick in range(L):
        p_root = p_rate*skel_pos_array[tick,0,:]
        R_offset_list = []
        for joint_idx in range(chain.get_n_joint()):
            if joint_idx == 0: # root 
                rpy_deg = np.array([0,0,0])
                rpy_deg_root = skel_rpy_deg_array[tick,joint_idx,:] # backup root rot
            else:
                rpy_deg = skel_rpy_deg_array[tick,joint_idx,:]
            R_offset_list.append(
                rpy2r_order(np.radians(rpy_deg),order=rpy_order)
            )
            
        R_root = rpy2r_order(np.radians(rpy_deg_root),order=rpy_order)
        chain.set_root_joint_R(R=R_root)
        chain.set_root_joint_p(p=p_root)
        
        chain.set_joint_R_offset_list(chain.joint_names,R_offset_list)
        chain.forward_kinematics()
        # Append
        chains.append(deepcopy(chain)) # we have to 'deepcopy'
    
    # Return
    return secs,chains,hand_tip_pos