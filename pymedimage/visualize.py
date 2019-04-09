"""
Visualization of 3d numpy array for debugging and visualization
Based on code from internet
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data#gs.v0bGR6M
"""
# TODO: Add orientation selection and automatic interpolation
import numpy as np
# NOTE: this is for debugging
import matplotlib.pyplot as plt
## used for ubuntu 14.04
#gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
#for gui in gui_env:
#    try:
#        matplotlib.use(gui,warn=False, force=True)
#        from matplotlib import pyplot as plt
#        break
#    except:
#        continue
#module=plt
#switch to Qt

def multi_slice_viewer(volume,_slice = None):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    if _slice is not None:
        _slice = int(_slice)
        if (_slice >= 0) and (_slice < volume.shape[0]):
            ax.index = int(_slice)
        else:
            raise ValueError("Invalid slice indexing %s out of %s"%(_slice,volume.shape[0]))
    else:
        ax.index = volume.shape[0] // 2
    ax.set_xlabel("Slice %s of %s"%(str(ax.index),volume.shape[0]))
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_scroll)
    plt.draw()
    plt.show()

def double_viewer(volume1, volume2 ,_slice = None):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.volume = volume1
    ax2.volume = volume2
    if _slice is not None:
        _slice = int(_slice)
        if (_slice >= 0) and (_slice < volume.shape[0]):
            ax1.index = int(_slice)
            ax2.index = int(_slice)
        else:
            raise ValueError("Invalid slice indexing %s out of %s"%(_slice,volume.shape[0]))
    else:
        ax1.index = volume1.shape[0] // 2
        ax2.index = volume2.shape[0] // 2
    ax1.set_xlabel("Slice %s of %s"%(str(ax1.index),volume1.shape[0]))
    ax2.set_xlabel("Slice %s of %s"%(str(ax2.index),volume2.shape[0]))
    ax1.imshow(volume1[ax1.index])
    ax2.imshow(volume2[ax2.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_scroll)
    plt.draw()
    plt.show()


def triple_viewer(volume1, volume2 , volume3,_slice = None):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.volume = volume1
    ax2.volume = volume2
    ax3.volume = volume3
    if _slice is not None:
        _slice = int(_slice)
        if (_slice >= 0) and (_slice < volume.shape[0]):
            ax1.index = int(_slice)
            ax2.index = int(_slice)
            ax3.index = int(_slice)
        else:
            raise ValueError("Invalid slice indexing %s out of %s"%(_slice,volume.shape[0]))
    else:
        ax1.index = volume1.shape[0] // 2
        ax2.index = volume2.shape[0] // 2
        ax3.index = volume3.shape[0] // 2
    ax1.set_xlabel("Slice %s of %s"%(str(ax1.index),volume1.shape[0]))
    ax2.set_xlabel("Slice %s of %s"%(str(ax2.index),volume2.shape[0]))
    ax3.set_xlabel("Slice %s of %s"%(str(ax3.index),volume3.shape[0]))
    ax1.imshow(volume1[ax1.index])
    ax2.imshow(volume2[ax2.index])
    ax3.imshow(volume3[ax3.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_scroll)
    plt.draw()
    plt.show()





def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.set_xlabel("Slice %s of %s"%(str(ax.index),volume.shape[0]))
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.set_xlabel("Slice %s of %s"%(str(ax.index),volume.shape[0]))
    ax.images[0].set_array(volume[ax.index])

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    ## added
    for ax in fig.axes:
    ##
        if event.button == 'up':
            previous_slice(ax)
        elif event.button == 'down':
            next_slice(ax)
        fig.canvas.draw()

